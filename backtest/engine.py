"""
Event-driven bar-by-bar backtester.

Modes:
  rules  -- signal_logic.generate_signal() only, no LLM
  llm    -- call Claude API for every bar (expensive)
  hybrid -- rules generate signal, LLM veto-only

Scale-out:
  50 % at TP1 -> SL -> breakeven
  30 % at TP2
  20 % at TP3

Performance note: indicators are computed ONCE on the full DataFrame;
the 4h bias is looked up per-bar from a pre-built 4h index; SMC features
(FVGs, OBs, liquidity) are computed only when we actually need to generate
a new entry signal, keeping the loop O(n).
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd

from indicators import compute_indicators
from signal_logic import generate_signal


# ── constants ──────────────────────────────────────────────────────────────────
FEE_PCT      = 0.0005   # 0.05% per side (taker)
SLIPPAGE_PCT = 0.0002   # 0.02%
TP1_FRAC     = 0.50
TP2_FRAC     = 0.30
TP3_FRAC     = 0.20


# ── SMC helpers (pure, no exchange) ───────────────────────────────────────────

def _find_fvgs(df: pd.DataFrame) -> list:
    fvgs = []
    arr = df[['high', 'low']].values
    n = len(arr)
    for i in range(1, n - 1):
        prev_high, nxt_low = arr[i - 1][0], arr[i + 1][1]
        if nxt_low > prev_high:
            fvgs.append({'type': 'bull', 'top': float(nxt_low), 'bot': float(prev_high)})
        prev_low, nxt_high = arr[i - 1][1], arr[i + 1][0]
        if nxt_high < prev_low:
            fvgs.append({'type': 'bear', 'top': float(prev_low), 'bot': float(nxt_high)})
    return fvgs[-4:]


def _find_obs(df: pd.DataFrame) -> list:
    obs = []
    closes = df['close'].values
    opens  = df['open'].values
    highs  = df['high'].values
    lows   = df['low'].values
    n = len(df)
    for i in range(5, n - 3):
        body = abs(closes[i] - opens[i]) / opens[i]
        if body > 0.004:
            move = (closes[min(i + 3, n - 1)] - closes[i]) / closes[i]
            if move > 0.008 and closes[i] < opens[i]:
                obs.append({'price': float(opens[i]), 'top': float(highs[i]),
                            'bot': float(lows[i]), 'type': 'bullish'})
            elif move < -0.008 and closes[i] > opens[i]:
                obs.append({'price': float(opens[i]), 'top': float(highs[i]),
                            'bot': float(lows[i]), 'type': 'bearish'})
    return obs[-3:]


def _find_liq(df: pd.DataFrame) -> dict:
    recent = df.tail(30)
    return {'bsl': recent['high'].nlargest(3).tolist(),
            'ssl': recent['low'].nsmallest(3).tolist()}


def _single_tf_bias(df_with_indicators: pd.DataFrame) -> str:
    """Compute BULLISH/BEARISH/NEUTRAL bias from a pre-computed indicator df."""
    if len(df_with_indicators) < 55:
        return 'NEUTRAL'
    df = df_with_indicators.dropna(subset=['ema20', 'ema50'])
    if df.empty:
        return 'NEUTRAL'
    last     = df.iloc[-1]
    lookback = min(5, len(df) - 1)
    ema20_prev = float(df.iloc[-(lookback + 1)]['ema20'])
    slope_up   = float(last['ema20']) > ema20_prev
    if last['close'] > last['ema20'] > last['ema50'] and slope_up:
        return 'BULLISH'
    if last['close'] < last['ema20'] < last['ema50'] and not slope_up:
        return 'BEARISH'
    return 'NEUTRAL'


def _build_4h_bias_series(df_1h_with_ind: pd.DataFrame) -> pd.Series:
    """
    Pre-compute the 4h bias for every 1h timestamp.
    Returns a pd.Series indexed the same as df_1h, values BULLISH/BEARISH/NEUTRAL.
    This is computed ONCE and then looked up O(1) per bar.
    """
    df_4h = df_1h_with_ind.resample('4H').agg(
        {'open': 'first', 'high': 'max', 'low': 'min',
         'close': 'last', 'volume': 'sum'}
    ).dropna()
    df_4h = compute_indicators(df_4h)

    # For each 4h bar, compute bias using all preceding 4h bars
    biases_4h: List[Tuple] = []
    for i in range(55, len(df_4h)):
        b = _single_tf_bias(df_4h.iloc[:i + 1])
        biases_4h.append((df_4h.index[i], b))

    if not biases_4h:
        return pd.Series('NEUTRAL', index=df_1h_with_ind.index)

    bias_series_4h = pd.Series(
        [b for _, b in biases_4h],
        index=[ts for ts, _ in biases_4h],
    )

    # Forward-fill to 1h frequency
    combined = bias_series_4h.reindex(
        df_1h_with_ind.index.union(bias_series_4h.index)
    ).ffill().reindex(df_1h_with_ind.index).fillna('NEUTRAL')
    return combined


def _fast_market_data(df: pd.DataFrame, df_4h_bias: pd.Series,
                      i: int, symbol: str) -> dict:
    """
    Build market_data dict at bar i using pre-computed indicator columns
    and 4h bias series. SMC features are computed on a trailing window.
    """
    row   = df.iloc[i]
    price = float(row['close'])
    atr   = float(row['atr'])   if not pd.isna(row.get('atr', float('nan'))) else 0.0
    ts    = df.index[i]

    b_1h = _single_tf_bias(df.iloc[max(0, i - 299): i + 1])
    b_4h = df_4h_bias.get(ts, 'NEUTRAL') if hasattr(df_4h_bias, 'get') else (
        df_4h_bias.iloc[i] if i < len(df_4h_bias) else 'NEUTRAL')
    # try proper lookup
    try:
        b_4h = df_4h_bias[ts]
    except (KeyError, IndexError):
        b_4h = 'NEUTRAL'

    b_15m = b_1h  # proxy
    biases = {'15m': b_15m, '1h': b_1h, '4h': b_4h}
    bulls  = sum(1 for v in biases.values() if v == 'BULLISH')
    bears  = sum(1 for v in biases.values() if v == 'BEARISH')
    overall  = 'BULLISH' if bulls >= 2 else ('BEARISH' if bears >= 2 else 'MIXED')
    tf_bias  = {'per_tf': biases, 'overall': overall, 'agreement': max(bulls, bears)}

    # SMC (on trailing 100 bars — only computed on potential entry bars)
    window = df.iloc[max(0, i - 99): i + 1]
    fvgs   = _find_fvgs(window)
    obs    = _find_obs(window)
    liq    = _find_liq(window)

    # BB width percentile
    bb_widths = (df['bb_upper'] - df['bb_lower']).iloc[max(0, i - 99): i + 1].dropna()
    cur_bbw   = float(row['bb_upper'] - row['bb_lower']) if not pd.isna(row.get('bb_upper', float('nan'))) else 0
    bb_pct    = int((bb_widths < cur_bbw).sum() / len(bb_widths) * 100) if len(bb_widths) > 1 else 50
    cur_adx   = float(row['adx']) if 'adx' in row.index and not pd.isna(row['adx']) else 25.0

    def _f(col, default=0.0):
        v = row.get(col, float('nan'))
        return default if pd.isna(v) else float(v)

    return {
        'symbol':              symbol,
        'price':               price,
        'atr':                 atr,
        'rsi':                 _f('rsi', 50.0),
        'macd':                _f('macd'),
        'macd_sig':            _f('macd_sig'),
        'ema20':               _f('ema20', price),
        'ema50':               _f('ema50', price),
        'ema200':              None if pd.isna(row.get('ema200', float('nan'))) else float(row['ema200']),
        'bb_upper':            _f('bb_upper', price),
        'bb_lower':            _f('bb_lower', price),
        'volume_signal':       'NORMAL',
        'volume_trend':        'FLAT',
        'rsi_divergence':      'NONE',
        'candle_pattern':      '',
        'timeframe_bias':      tf_bias,
        'fear_greed':          {'value': 50, 'label': 'Neutral'},
        'liquidity':           liq,
        'order_blocks':        obs,
        'fvgs':                fvgs,
        'adx':                 cur_adx,
        'bb_width_percentile': bb_pct,
    }


# ── Trade dataclass ────────────────────────────────────────────────────────────

@dataclass
class Trade:
    symbol:        str
    action:        str
    entry:         float
    sl:            float
    tp1:           float
    tp2:           float
    tp3:           float
    confidence:    int
    position_usdt: float
    entry_bar:     int
    entry_time:    object = None

    tp1_hit:      bool  = False
    tp2_hit:      bool  = False
    realized_pnl: float = 0.0

    def qty_remaining_frac(self) -> float:
        frac = 1.0
        if self.tp1_hit:
            frac -= TP1_FRAC
        if self.tp2_hit:
            frac -= TP2_FRAC
        return frac


# ── PnL helper ─────────────────────────────────────────────────────────────────

def _fill_pnl(action: str, entry: float, exit_price: float,
              frac: float, position_usdt: float) -> float:
    partial_usdt = position_usdt * frac
    raw_ret = ((exit_price - entry) / entry if action == 'LONG'
               else (entry - exit_price) / entry)
    cost = 2 * (FEE_PCT + SLIPPAGE_PCT)
    return partial_usdt * (raw_ret - cost)


# ── Exit checker ───────────────────────────────────────────────────────────────

def _check_exits(trade: Trade, bar: pd.Series) -> Tuple[bool, Optional[dict]]:
    hi, lo = bar['high'], bar['low']
    action = trade.action

    def _close_all(exit_price, reason):
        pnl = _fill_pnl(action, trade.entry, exit_price,
                        trade.qty_remaining_frac(), trade.position_usdt)
        trade.realized_pnl += pnl
        return True, {'exit_price': exit_price, 'exit_reason': reason,
                      'pnl_usdt': trade.realized_pnl,
                      'confidence': trade.confidence, 'exit_time': bar.name}

    # SL
    if action == 'LONG' and lo <= trade.sl:
        return _close_all(trade.sl, 'SL_HIT')
    if action == 'SHORT' and hi >= trade.sl:
        return _close_all(trade.sl, 'SL_HIT')

    # TP1
    if not trade.tp1_hit:
        if (action == 'LONG' and hi >= trade.tp1) or (action == 'SHORT' and lo <= trade.tp1):
            pnl = _fill_pnl(action, trade.entry, trade.tp1, TP1_FRAC, trade.position_usdt)
            trade.realized_pnl += pnl
            trade.tp1_hit = True
            trade.sl = trade.entry   # move to BE

    # TP2
    if trade.tp1_hit and not trade.tp2_hit:
        if (action == 'LONG' and hi >= trade.tp2) or (action == 'SHORT' and lo <= trade.tp2):
            pnl = _fill_pnl(action, trade.entry, trade.tp2, TP2_FRAC, trade.position_usdt)
            trade.realized_pnl += pnl
            trade.tp2_hit = True

    # TP3
    if trade.tp1_hit and trade.tp2_hit:
        if (action == 'LONG' and hi >= trade.tp3) or (action == 'SHORT' and lo <= trade.tp3):
            pnl = _fill_pnl(action, trade.entry, trade.tp3, TP3_FRAC, trade.position_usdt)
            trade.realized_pnl += pnl
            return True, {'exit_price': trade.tp3, 'exit_reason': 'TP3_HIT',
                          'pnl_usdt': trade.realized_pnl,
                          'confidence': trade.confidence, 'exit_time': bar.name}

    return False, None


# ── Main engine ────────────────────────────────────────────────────────────────

class BarReplayer:
    """
    Bar-by-bar event-driven backtester.

    Parameters
    ----------
    symbol         : e.g. 'BTC/USDT'
    initial_capital: Starting USDT balance
    risk_pct       : Percent of capital risked per trade
    mode           : 'rules' | 'llm' | 'hybrid'
    warmup         : Bars to skip while indicators warm up
    min_confidence : Minimum rule-score to enter a trade
    """

    def __init__(
        self,
        symbol: str = 'BTC/USDT',
        timeframe: str = '1h',
        initial_capital: float = 10_000.0,
        risk_pct: float = 1.0,
        mode: str = 'rules',
        warmup: int = 220,
        min_confidence: int = 65,
        agent=None,
    ):
        self.symbol   = symbol
        self.tf       = timeframe
        self.initial  = initial_capital
        self.risk_pct = risk_pct
        self.mode     = mode
        self.warmup   = warmup
        self.min_conf = min_confidence
        self.agent    = agent

    def _position_usdt(self, capital: float, entry: float, sl: float) -> float:
        risk_usdt   = capital * self.risk_pct / 100
        sl_dist_pct = abs(entry - sl) / entry
        if sl_dist_pct <= 0:
            return 0.0
        return min(risk_usdt / sl_dist_pct, capital * 10)

    def _get_signal(self, market_data: dict) -> dict:
        if self.mode == 'rules':
            return generate_signal(market_data, min_confidence=self.min_conf)
        if self.mode == 'llm' and self.agent:
            try:
                return self.agent.get_signal(market_data)
            except Exception as e:
                return {'action': 'WAIT', 'reason': str(e)}
        if self.mode == 'hybrid' and self.agent:
            sig = generate_signal(market_data, min_confidence=self.min_conf)
            if sig['action'] == 'WAIT':
                return sig
            try:
                veto = self.agent.veto_signal(sig, market_data)
                if veto.get('decision') == 'REJECT':
                    return {'action': 'WAIT', 'reason': 'LLM veto: ' + veto.get('reason', '')}
                adj = veto.get('confidence_adjustment', 0)
                sig['confidence'] = max(0, min(100, sig['confidence'] + adj))
                if sig['confidence'] < self.min_conf:
                    return {'action': 'WAIT', 'reason': f'Post-veto conf low'}
            except Exception:
                pass
            return sig
        return generate_signal(market_data, min_confidence=self.min_conf)

    # also expose as static for tests
    @staticmethod
    def _check_exits(trade: Trade, bar: pd.Series) -> Tuple[bool, Optional[dict]]:
        return _check_exits(trade, bar)

    def run(self, df: pd.DataFrame) -> Tuple[list, list]:
        """Run the backtest. Returns (trades, equity_curve)."""
        # Pre-compute all indicators ONCE
        print(f'[Backtest] Computing indicators for {len(df)} bars ...')
        df = compute_indicators(df)
        df = df.dropna(subset=['ema20', 'ema50', 'rsi', 'atr', 'macd'])
        print(f'[Backtest] {len(df)} usable bars after warm-up. Building 4h bias series ...')

        # Pre-compute 4h bias for every 1h bar (done once, O(n) lookup later)
        df_4h_bias = _build_4h_bias_series(df)
        print(f'[Backtest] Starting bar replay ({len(df) - self.warmup} bars) ...')

        capital:    float          = self.initial
        equity:     list           = [capital]
        trades:     list           = []
        open_trade: Optional[Trade] = None

        for i in range(self.warmup, len(df)):
            bar = df.iloc[i]

            if open_trade is not None:
                closed, exit_info = _check_exits(open_trade, bar)
                if closed:
                    capital += open_trade.realized_pnl
                    trades.append({
                        'symbol':        open_trade.symbol,
                        'action':        open_trade.action,
                        'entry':         open_trade.entry,
                        'entry_time':    str(open_trade.entry_time),
                        'exit_time':     str(exit_info['exit_time']),
                        'exit_price':    exit_info['exit_price'],
                        'exit_reason':   exit_info['exit_reason'],
                        'pnl_usdt':      round(open_trade.realized_pnl, 4),
                        'confidence':    open_trade.confidence,
                        'tp1':           open_trade.tp1,
                        'sl':            open_trade.sl,
                        'capital_after': round(capital, 2),
                    })
                    open_trade = None

            equity.append(capital)

            if open_trade is not None or capital <= 0:
                continue

            try:
                market_data = _fast_market_data(df, df_4h_bias, i, self.symbol)
            except Exception:
                continue

            signal = self._get_signal(market_data)
            if signal.get('action') not in ('LONG', 'SHORT'):
                continue

            entry_price = float(bar['close'])
            slip = SLIPPAGE_PCT
            actual_entry = (entry_price * (1 + slip) if signal['action'] == 'LONG'
                            else entry_price * (1 - slip))

            pos_usdt = self._position_usdt(capital, actual_entry, signal['sl'])
            if pos_usdt <= 0:
                continue

            open_trade = Trade(
                symbol=self.symbol, action=signal['action'],
                entry=actual_entry, sl=signal['sl'],
                tp1=signal['tp1'], tp2=signal['tp2'], tp3=signal['tp3'],
                confidence=signal['confidence'], position_usdt=pos_usdt,
                entry_bar=i, entry_time=bar.name,
            )

        # Close any open trade at end of data
        if open_trade is not None:
            last = df.iloc[-1]
            ep   = float(last['close'])
            pnl  = _fill_pnl(open_trade.action, open_trade.entry, ep,
                              open_trade.qty_remaining_frac(), open_trade.position_usdt)
            open_trade.realized_pnl += pnl
            capital += open_trade.realized_pnl
            trades.append({
                'symbol': open_trade.symbol, 'action': open_trade.action,
                'entry': open_trade.entry, 'entry_time': str(open_trade.entry_time),
                'exit_time': str(last.name), 'exit_price': ep,
                'exit_reason': 'END_OF_DATA',
                'pnl_usdt': round(open_trade.realized_pnl, 4),
                'confidence': open_trade.confidence, 'tp1': open_trade.tp1,
                'sl': open_trade.sl, 'capital_after': round(capital, 2),
            })
            equity.append(capital)

        print(f'[Backtest] Done. {len(trades)} trades closed.')
        return trades, equity

    def walk_forward(
        self, df: pd.DataFrame,
        train_months: int = 6,
        test_months:  int = 2,
    ) -> Tuple[list, list]:
        """Rolling walk-forward: train_months in-sample, test_months out-of-sample."""
        df = compute_indicators(df)
        df = df.dropna(subset=['ema20', 'ema50', 'rsi', 'atr', 'macd'])

        bars_per_month = 30 * 24
        train_len  = train_months * bars_per_month
        test_len   = test_months  * bars_per_month
        window_len = train_len + test_len

        all_trades: list = []
        oos_equity: list = [self.initial]
        capital = self.initial
        window_num = 0
        start = 0

        while start + window_len <= len(df):
            oos_start = start + train_len
            oos_end   = start + window_len
            df_oos    = df.iloc[oos_start:oos_end]

            replayer = BarReplayer(
                symbol=self.symbol, timeframe=self.tf,
                initial_capital=capital, risk_pct=self.risk_pct,
                mode=self.mode, warmup=self.warmup,
                min_confidence=self.min_conf, agent=self.agent,
            )
            wt, weq = replayer.run(df_oos)
            for t in wt:
                t['walk_forward_window'] = window_num
            all_trades.extend(wt)

            if len(weq) > 1:
                ratio = capital / self.initial if self.initial > 0 else 1.0
                oos_equity.extend(e * ratio for e in weq[1:])
                capital = weq[-1] * ratio

            start += test_len
            window_num += 1
            if df_oos.index is not None and len(df_oos):
                print(f'[WalkFwd] window {window_num}: '
                      f'{df_oos.index[0].date()} to {df_oos.index[-1].date()} '
                      f'| {len(wt)} trades')

        return all_trades, oos_equity
