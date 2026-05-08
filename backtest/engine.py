"""
Event-driven bar-by-bar backtester.

Modes:
  rules  — signal_logic.generate_signal() only, no LLM
  llm    — call Claude API for every bar (expensive)
  hybrid — rules generate signal, LLM veto-only

Scale-out:
  50 % at TP1 → SL → breakeven
  30 % at TP2
  20 % at TP3
"""
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import pandas as pd

from indicators import compute_indicators
from signal_logic import generate_signal, score_confidence


# ── constants ──────────────────────────────────────────────────────────────────
FEE_PCT      = 0.0005   # 0.05% per side (taker)
SLIPPAGE_PCT = 0.0002   # 0.02%
TP1_FRAC     = 0.50
TP2_FRAC     = 0.30
TP3_FRAC     = 0.20


# ── SMC helpers (pure, no exchange) ───────────────────────────────────────────

def _find_fvgs(df: pd.DataFrame) -> list:
    fvgs = []
    for i in range(1, len(df) - 1):
        prev, nxt = df.iloc[i - 1], df.iloc[i + 1]
        sub = df.iloc[i + 2:]
        if nxt['low'] > prev['high']:
            top, bot = float(nxt['low']), float(prev['high'])
            filled = any(sc['low'] <= bot for _, sc in sub.iterrows())
            if not filled:
                fvgs.append({'type': 'bull', 'top': top, 'bot': bot})
        if nxt['high'] < prev['low']:
            top, bot = float(prev['low']), float(nxt['high'])
            filled = any(sc['high'] >= top for _, sc in sub.iterrows())
            if not filled:
                fvgs.append({'type': 'bear', 'top': top, 'bot': bot})
    return fvgs[-4:]


def _find_obs(df: pd.DataFrame) -> list:
    obs = []
    for i in range(5, len(df) - 3):
        c = df.iloc[i]
        body = abs(c['close'] - c['open']) / c['open']
        if body > 0.004:
            move = (df.iloc[i + 3]['close'] - c['close']) / c['close']
            if move > 0.008 and c['close'] < c['open']:
                obs.append({'price': float(c['open']), 'top': float(c['high']),
                            'bot': float(c['low']), 'type': 'bullish'})
            elif move < -0.008 and c['close'] > c['open']:
                obs.append({'price': float(c['open']), 'top': float(c['high']),
                            'bot': float(c['low']), 'type': 'bearish'})
    return obs[-3:]


def _find_liq(df: pd.DataFrame) -> dict:
    recent = df.tail(30)
    return {'bsl': recent['high'].nlargest(3).tolist(),
            'ssl': recent['low'].nsmallest(3).tolist()}


def _tf_bias_from_df(df: pd.DataFrame) -> str:
    if len(df) < 55:
        return 'NEUTRAL'
    df = df.dropna(subset=['ema20', 'ema50'])
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


def _build_mtf_bias(df_1h: pd.DataFrame) -> dict:
    """Derive 15m / 1h / 4h biases from 1h data by resampling."""
    b1h = _tf_bias_from_df(df_1h)

    try:
        df_4h = df_1h.resample('4H').agg(
            {'open': 'first', 'high': 'max', 'low': 'min',
             'close': 'last', 'volume': 'sum'}
        ).dropna()
        df_4h = compute_indicators(df_4h)
        b4h = _tf_bias_from_df(df_4h)
    except Exception:
        b4h = 'NEUTRAL'

    b15m = b1h   # 15m not available in 1h backtest; use 1h as proxy

    biases = {'15m': b15m, '1h': b1h, '4h': b4h}
    bulls = sum(1 for v in biases.values() if v == 'BULLISH')
    bears = sum(1 for v in biases.values() if v == 'BEARISH')
    overall = 'BULLISH' if bulls >= 2 else ('BEARISH' if bears >= 2 else 'MIXED')
    return {'per_tf': biases, 'overall': overall, 'agreement': max(bulls, bears)}


def _build_market_data(df_window: pd.DataFrame, symbol: str = '') -> dict:
    """Build the market_data dict from a historical window of indicator-enriched data."""
    last = df_window.iloc[-1]
    price = float(last['close'])
    atr   = float(last['atr']) if not pd.isna(last.get('atr', float('nan'))) else 0.0

    tf_bias = _build_mtf_bias(df_window)
    fvgs    = _find_fvgs(df_window.tail(100))
    obs     = _find_obs(df_window)
    liq     = _find_liq(df_window)

    # BB width percentile (last 100 bars)
    bb_widths = (df_window['bb_upper'] - df_window['bb_lower']).tail(100).dropna()
    cur_bbw   = float(last['bb_upper'] - last['bb_lower']) if not pd.isna(last.get('bb_upper', float('nan'))) else 0
    bb_pct    = int((bb_widths < cur_bbw).sum() / len(bb_widths) * 100) if len(bb_widths) > 1 else 50
    cur_adx   = float(last['adx']) if 'adx' in last.index and not pd.isna(last['adx']) else 25.0

    return {
        'symbol':              symbol,
        'price':               price,
        'atr':                 atr,
        'rsi':                 float(last['rsi'])     if not pd.isna(last.get('rsi',     float('nan'))) else 50.0,
        'macd':                float(last['macd'])    if not pd.isna(last.get('macd',    float('nan'))) else 0.0,
        'macd_sig':            float(last['macd_sig']) if not pd.isna(last.get('macd_sig', float('nan'))) else 0.0,
        'ema20':               float(last['ema20'])   if not pd.isna(last.get('ema20',   float('nan'))) else price,
        'ema50':               float(last['ema50'])   if not pd.isna(last.get('ema50',   float('nan'))) else price,
        'ema200':              None if pd.isna(last.get('ema200', float('nan'))) else float(last['ema200']),
        'bb_upper':            float(last['bb_upper']) if not pd.isna(last.get('bb_upper', float('nan'))) else price,
        'bb_lower':            float(last['bb_lower']) if not pd.isna(last.get('bb_lower', float('nan'))) else price,
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
    symbol:       str
    action:       str
    entry:        float
    sl:           float
    tp1:          float
    tp2:          float
    tp3:          float
    confidence:   int
    position_usdt: float      # full position value in USDT
    entry_bar:    int
    entry_time:   object = None

    tp1_hit:      bool  = False
    tp2_hit:      bool  = False
    realized_pnl: float = 0.0

    def qty_remaining_frac(self) -> float:
        """Remaining fraction of original position."""
        frac = 1.0
        if self.tp1_hit:
            frac -= TP1_FRAC
        if self.tp2_hit:
            frac -= TP2_FRAC
        return frac


# ── PnL helper ─────────────────────────────────────────────────────────────────

def _fill_pnl(action: str, entry: float, exit_price: float,
              frac: float, position_usdt: float) -> float:
    """Net PnL for a partial or full close, after fees + slippage (round-trip)."""
    partial_usdt = position_usdt * frac
    if action == 'LONG':
        raw_ret = (exit_price - entry) / entry
    else:
        raw_ret = (entry - exit_price) / entry
    cost = 2 * (FEE_PCT + SLIPPAGE_PCT)
    return partial_usdt * (raw_ret - cost)


# ── Main engine ────────────────────────────────────────────────────────────────

class BarReplayer:
    """
    Bar-by-bar event-driven backtester.

    Parameters
    ----------
    symbol         : Trading pair, e.g. 'BTC/USDT'
    timeframe      : '1h' (only value supported in the backtester today)
    initial_capital: Starting USDT balance
    risk_pct       : Percent of capital risked per trade
    mode           : 'rules' | 'llm' | 'hybrid'
    warmup         : Number of bars to skip at the start (for indicator warm-up)
    min_confidence : Minimum rule-score to enter a trade (rules / hybrid mode)
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
        self.agent    = agent   # ClaudeAgent instance (llm / hybrid modes)

    # ── helpers ────────────────────────────────────────────────────────────────

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
                print(f'[Backtest] LLM call failed: {e}')
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
                    return {'action': 'WAIT', 'reason': f'Post-veto conf {sig["confidence"]} < {self.min_conf}'}
            except Exception as e:
                print(f'[Backtest] LLM veto failed: {e}')
            return sig

        return generate_signal(market_data, min_confidence=self.min_conf)

    @staticmethod
    def _check_exits(trade: Trade, bar: pd.Series) -> Tuple[bool, Optional[dict]]:
        """
        Check a bar against the trade's exit levels.
        Returns (trade_closed, exit_info_or_None).
        Partial closes update trade in-place and return (False, None).
        """
        hi, lo = bar['high'], bar['low']
        action = trade.action

        # SL check (highest priority)
        if action == 'LONG' and lo <= trade.sl:
            pnl = _fill_pnl(action, trade.entry, trade.sl,
                            trade.qty_remaining_frac(), trade.position_usdt)
            trade.realized_pnl += pnl
            return True, {
                'exit_price': trade.sl, 'exit_reason': 'SL_HIT',
                'pnl_usdt': trade.realized_pnl, 'confidence': trade.confidence,
                'exit_time': bar.name,
            }
        if action == 'SHORT' and hi >= trade.sl:
            pnl = _fill_pnl(action, trade.entry, trade.sl,
                            trade.qty_remaining_frac(), trade.position_usdt)
            trade.realized_pnl += pnl
            return True, {
                'exit_price': trade.sl, 'exit_reason': 'SL_HIT',
                'pnl_usdt': trade.realized_pnl, 'confidence': trade.confidence,
                'exit_time': bar.name,
            }

        # TP1
        if not trade.tp1_hit:
            tp1_hit = (action == 'LONG' and hi >= trade.tp1) or \
                      (action == 'SHORT' and lo <= trade.tp1)
            if tp1_hit:
                pnl = _fill_pnl(action, trade.entry, trade.tp1,
                                TP1_FRAC, trade.position_usdt)
                trade.realized_pnl += pnl
                trade.tp1_hit = True
                trade.sl = trade.entry   # move SL to breakeven

        # TP2
        if trade.tp1_hit and not trade.tp2_hit:
            tp2_hit = (action == 'LONG' and hi >= trade.tp2) or \
                      (action == 'SHORT' and lo <= trade.tp2)
            if tp2_hit:
                pnl = _fill_pnl(action, trade.entry, trade.tp2,
                                TP2_FRAC, trade.position_usdt)
                trade.realized_pnl += pnl
                trade.tp2_hit = True

        # TP3 — full close of remaining 20 %
        if trade.tp1_hit and trade.tp2_hit:
            tp3_hit = (action == 'LONG' and hi >= trade.tp3) or \
                      (action == 'SHORT' and lo <= trade.tp3)
            if tp3_hit:
                pnl = _fill_pnl(action, trade.entry, trade.tp3,
                                TP3_FRAC, trade.position_usdt)
                trade.realized_pnl += pnl
                return True, {
                    'exit_price': trade.tp3, 'exit_reason': 'TP3_HIT',
                    'pnl_usdt': trade.realized_pnl, 'confidence': trade.confidence,
                    'exit_time': bar.name,
                }

        return False, None

    # ── main loop ──────────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame) -> Tuple[list, list]:
        """
        Run the backtest on a pre-loaded indicator-enriched DataFrame.
        Returns (trades, equity_curve).
        """
        df = compute_indicators(df)
        df = df.dropna(subset=['ema20', 'ema50', 'rsi', 'atr', 'macd'])

        capital     = self.initial
        equity      = [capital]
        trades      = []
        open_trade: Optional[Trade] = None

        for i in range(self.warmup, len(df)):
            bar = df.iloc[i]

            # 1. Process open trade exits
            if open_trade is not None:
                closed, exit_info = self._check_exits(open_trade, bar)
                if closed:
                    capital += open_trade.realized_pnl
                    rec = {
                        'symbol':      open_trade.symbol,
                        'action':      open_trade.action,
                        'entry':       open_trade.entry,
                        'entry_time':  str(open_trade.entry_time),
                        'exit_time':   str(exit_info['exit_time']),
                        'exit_price':  exit_info['exit_price'],
                        'exit_reason': exit_info['exit_reason'],
                        'pnl_usdt':    round(open_trade.realized_pnl, 4),
                        'confidence':  open_trade.confidence,
                        'tp1':         open_trade.tp1,
                        'sl':          open_trade.sl,
                        'capital_after': round(capital, 2),
                    }
                    trades.append(rec)
                    open_trade = None

            equity.append(capital)

            # 2. Check for new entry (only one trade at a time)
            if open_trade is not None:
                continue
            if capital <= 0:
                break

            window = df.iloc[max(0, i - 299): i + 1]
            try:
                market_data = _build_market_data(window, self.symbol)
            except Exception:
                continue

            signal = self._get_signal(market_data)
            if signal.get('action') not in ('LONG', 'SHORT'):
                continue

            entry_price = float(bar['close'])
            if signal['action'] == 'LONG':
                actual_entry = entry_price * (1 + SLIPPAGE_PCT)
            else:
                actual_entry = entry_price * (1 - SLIPPAGE_PCT)

            pos_usdt = self._position_usdt(capital, actual_entry, signal['sl'])
            if pos_usdt <= 0:
                continue

            open_trade = Trade(
                symbol        = self.symbol,
                action        = signal['action'],
                entry         = actual_entry,
                sl            = signal['sl'],
                tp1           = signal['tp1'],
                tp2           = signal['tp2'],
                tp3           = signal['tp3'],
                confidence    = signal['confidence'],
                position_usdt = pos_usdt,
                entry_bar     = i,
                entry_time    = bar.name,
            )

        # Close any remaining open trade at last bar
        if open_trade is not None:
            last_bar = df.iloc[-1]
            exit_price = float(last_bar['close'])
            pnl = _fill_pnl(open_trade.action, open_trade.entry, exit_price,
                            open_trade.qty_remaining_frac(), open_trade.position_usdt)
            open_trade.realized_pnl += pnl
            capital += open_trade.realized_pnl
            trades.append({
                'symbol':        open_trade.symbol,
                'action':        open_trade.action,
                'entry':         open_trade.entry,
                'entry_time':    str(open_trade.entry_time),
                'exit_time':     str(last_bar.name),
                'exit_price':    exit_price,
                'exit_reason':   'END_OF_DATA',
                'pnl_usdt':      round(open_trade.realized_pnl, 4),
                'confidence':    open_trade.confidence,
                'tp1':           open_trade.tp1,
                'sl':            open_trade.sl,
                'capital_after': round(capital, 2),
            })
            equity.append(capital)

        return trades, equity

    # ── walk-forward ───────────────────────────────────────────────────────────

    def walk_forward(
        self, df: pd.DataFrame,
        train_months: int = 6,
        test_months: int  = 2,
    ) -> Tuple[list, list]:
        """
        Rolling walk-forward: train_months in-sample, test_months out-of-sample.
        Returns (all_oos_trades, oos_equity) concatenated across all windows.
        """
        df = compute_indicators(df)
        df = df.dropna(subset=['ema20', 'ema50', 'rsi', 'atr', 'macd'])

        bars_per_month = 30 * 24   # 1h bars
        train_len = train_months * bars_per_month
        test_len  = test_months  * bars_per_month
        window_len = train_len + test_len

        all_trades: list = []
        oos_equity: list = [self.initial]
        capital = self.initial

        start = 0
        window_num = 0
        while start + window_len <= len(df):
            oos_start = start + train_len
            oos_end   = start + window_len
            df_oos    = df.iloc[oos_start:oos_end]

            replayer = BarReplayer(
                symbol          = self.symbol,
                timeframe       = self.tf,
                initial_capital = capital,
                risk_pct        = self.risk_pct,
                mode            = self.mode,
                warmup          = self.warmup,
                min_confidence  = self.min_conf,
                agent           = self.agent,
            )
            window_trades, window_eq = replayer.run(df_oos)

            for t in window_trades:
                t['walk_forward_window'] = window_num
            all_trades.extend(window_trades)

            if len(window_eq) > 1:
                # Chain equity curves: scale by ratio from last known capital
                ratio = capital / self.initial if self.initial > 0 else 1.0
                oos_equity.extend(e * ratio for e in window_eq[1:])
                capital = window_eq[-1] * ratio

            start += test_len
            window_num += 1
            print(f'[WalkForward] window {window_num}: '
                  f'{df_oos.index[0].date()} → {df_oos.index[-1].date()} '
                  f'| {len(window_trades)} trades')

        return all_trades, oos_equity
