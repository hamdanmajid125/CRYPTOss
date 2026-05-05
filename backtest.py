"""
Backtesting engine — mirrors live signal logic without calling the Claude API.
Uses real OHLCV data via DataFeed, same indicators, same confidence scorer.
"""
import json
import math
import time
import numpy as np
import pandas as pd
from typing import Dict, List, Optional

from data_feed import DataFeed


class Backtester:

    # Candles per year for Sharpe annualisation
    _CPY = {
        '1m': 525_600, '3m': 175_200, '5m': 105_120, '15m': 35_040,
        '30m': 17_520, '1h': 8_760, '2h': 4_380, '4h': 2_190,
        '6h': 1_460, '12h': 730, '1d': 365,
    }

    def __init__(self, feed: DataFeed, symbol: str, timeframe: str = '1h',
                 initial_capital: float = 10_000.0, risk_pct: float = 1.5):
        self.feed = feed
        self.symbol = symbol
        self.timeframe = timeframe
        self.initial_capital = float(initial_capital)
        self.risk_pct = float(risk_pct)

        # Import scorer — _score_confidence makes zero API calls
        from claude_agent import ClaudeAgent
        self._scorer = ClaudeAgent(api_key='backtest_no_api')

    # ── HELPERS ────────────────────────────────────────────────────────────────

    def _bias(self, df: pd.DataFrame) -> dict:
        """Single-TF bias mirroring get_timeframe_bias() EMA + RSI logic."""
        last  = df.iloc[-1]
        close = float(last['close'])
        ema20 = float(last['ema20']) if not pd.isna(last['ema20']) else close
        ema50 = float(last['ema50']) if not pd.isna(last['ema50']) else close
        rsi   = float(last['rsi'])   if not pd.isna(last['rsi'])   else 50.0

        if close > ema20 > ema50 and rsi > 50:
            overall, agreement = 'BULLISH', 2
        elif close < ema20 < ema50 and rsi < 50:
            overall, agreement = 'BEARISH', 2
        else:
            overall, agreement = 'MIXED', 1

        return {'overall': overall, 'agreement': agreement,
                'per_tf': {self.timeframe: overall}}

    def _vol_signal(self, df: pd.DataFrame) -> str:
        last = df.iloc[-1]
        vsma = last.get('volume_sma', None)
        if vsma is None or pd.isna(vsma) or vsma == 0:
            return 'NORMAL'
        r = last['volume'] / vsma
        if r >= 2.0: return 'VERY HIGH — strong confirmation'
        if r >= 1.5: return 'HIGH — confirms direction'
        if r <= 0.5: return 'VERY LOW — weak, avoid trading'
        if r <= 0.7: return 'LOW — signal is weaker'
        return 'NORMAL'

    def _vol_trend(self, df: pd.DataFrame) -> str:
        v = df['volume'].tail(5).values
        if len(v) < 5:           return 'FLAT'
        if v[-1] > v[0] * 1.3:  return 'INCREASING — momentum building'
        if v[-1] < v[0] * 0.7:  return 'DECREASING — momentum fading'
        return 'FLAT'

    def _build_data(self, hist: pd.DataFrame) -> Optional[dict]:
        """Build the data dict required by _score_confidence."""
        last = hist.iloc[-1]
        for col in ('ema20', 'ema50', 'rsi', 'atr', 'macd', 'macd_sig'):
            if col not in hist.columns or pd.isna(last[col]):
                return None

        # Cap expensive SMC lookups to last 100 candles to keep backtest fast
        pat_win = hist.tail(100)
        price   = float(last['close'])

        return {
            'symbol':         self.symbol,
            'price':          price,
            'rsi':            float(last['rsi']),
            'macd':           float(last['macd']),
            'macd_sig':       float(last['macd_sig']),
            'atr':            float(last['atr']),
            'ema20':          float(last['ema20']),
            'ema50':          float(last['ema50']),
            'ema200':         float(last['ema200']) if not pd.isna(last['ema200']) else None,
            'volume_signal':  self._vol_signal(hist),
            'volume_trend':   self._vol_trend(hist),
            'rsi_divergence': self.feed.detect_rsi_divergence(hist),
            'candle_pattern': self.feed.detect_candle_pattern(hist),
            'fvgs':           self.feed.find_fvgs(pat_win),
            'order_blocks':   self.feed.find_order_blocks(pat_win),
            'fear_greed':     {'value': 50, 'label': 'Neutral'},
        }

    def _get_signal(self, data: dict, tf_bias: dict) -> str:
        """Rule-based signal matching Claude's live decision logic (no API)."""
        overall = tf_bias.get('overall', 'MIXED')
        if overall == 'MIXED':
            return 'WAIT'

        rsi   = data.get('rsi', 50)
        price = data.get('price', 0)
        ema50 = data.get('ema50', price)
        fng   = data['fear_greed']
        conf  = self._scorer._score_confidence(data, tf_bias, fng)

        if overall == 'BULLISH' and conf >= 65 and rsi < 70 and price > ema50:
            return 'LONG'
        if overall == 'BEARISH' and conf >= 65 and rsi > 30 and price < ema50:
            return 'SHORT'
        return 'WAIT'

    # ── MAIN RUN ───────────────────────────────────────────────────────────────

    def run(self, lookback_candles: int = 500) -> dict:
        t0 = time.time()
        print(f'[Backtest] {self.symbol} {self.timeframe} — '
              f'fetching {lookback_candles} candles...')

        df = self.feed.get_ohlcv(self.symbol, self.timeframe,
                                 limit=lookback_candles)
        df = self.feed.add_indicators(df)
        n  = len(df)

        WARMUP   = 60   # candles before simulation starts (indicator warmup)
        MAX_HOLD = 48   # force-close after this many candles
        SL_ATR   = 1.5
        TP_ATR   = 2.0

        trades:      List[Dict] = []
        capital      = self.initial_capital
        equity_curve = [capital]
        open_trade:  Optional[Dict] = None

        print(f'[Backtest] Walking {max(0, n - WARMUP - 1)} candles...')

        for i in range(WARMUP, n - 1):
            hist    = df.iloc[:i + 1]
            current = hist.iloc[-1]
            nxt     = df.iloc[i + 1]   # entry = nxt['open'] (no lookahead)

            # Skip candles where core indicators are still NaN
            if any(pd.isna(current.get(c, float('nan')))
                   for c in ('ema20', 'ema50', 'rsi', 'atr', 'macd', 'macd_sig')):
                equity_curve.append(capital)
                continue

            # ── Check open trade on this candle ───────────────────────────────
            if open_trade is not None:
                hi   = float(current['high'])
                lo   = float(current['low'])
                side = open_trade['side']
                ep   = open_trade['entry']
                sl_p = open_trade['sl']
                tp_p = open_trade['tp1']
                pos  = open_trade['position_usdt']

                hit_sl = (lo <= sl_p) if side == 'LONG' else (hi >= sl_p)
                hit_tp = (hi >= tp_p) if side == 'LONG' else (lo <= tp_p)
                timed  = open_trade['hold'] >= MAX_HOLD

                # Conservative assumption: if both hit same candle, SL wins
                if hit_sl and hit_tp:
                    hit_tp = False

                if hit_sl or hit_tp or timed:
                    if hit_tp:
                        xp = tp_p; xr = 'TP'
                    elif hit_sl:
                        xp = sl_p; xr = 'SL'
                    else:
                        xp = float(current['close']); xr = 'TIMEOUT'

                    pnl = (pos * (xp - ep) / ep if side == 'LONG'
                           else pos * (ep - xp) / ep)
                    capital += pnl

                    trades.append({
                        'side':        side,
                        'entry':       round(ep, 8),
                        'exit':        round(xp, 8),
                        'sl':          round(sl_p, 8),
                        'tp1':         round(tp_p, 8),
                        'pnl_usdt':    round(pnl, 4),
                        'pnl_pct':     round(pnl / self.initial_capital * 100, 4),
                        'hold':        open_trade['hold'],
                        'outcome':     'WIN' if pnl > 0 else 'LOSS',
                        'exit_reason': xr,
                    })
                    open_trade = None
                else:
                    open_trade['hold'] += 1

            equity_curve.append(capital)

            # ── Generate new signal if flat ────────────────────────────────────
            if open_trade is None:
                data = self._build_data(hist)
                if data is None:
                    continue

                tf_bias = self._bias(hist)
                sig     = self._get_signal(data, tf_bias)

                if sig in ('LONG', 'SHORT'):
                    entry_p = float(nxt['open'])
                    atr     = float(current['atr'])

                    if entry_p <= 0 or atr <= 0:
                        continue

                    sl_p = entry_p - atr * SL_ATR if sig == 'LONG' else entry_p + atr * SL_ATR
                    tp_p = entry_p + atr * TP_ATR if sig == 'LONG' else entry_p - atr * TP_ATR

                    sl_pct = abs(entry_p - sl_p) / entry_p
                    if sl_pct <= 0:
                        continue

                    risk_usd = capital * (self.risk_pct / 100)
                    pos_usd  = risk_usd / sl_pct

                    open_trade = {
                        'side':          sig,
                        'entry':         entry_p,
                        'sl':            sl_p,
                        'tp1':           tp_p,
                        'position_usdt': pos_usd,
                        'hold':          0,
                    }

        # ── Force-close any trade still open at end of data ────────────────────
        if open_trade is not None:
            xp   = float(df.iloc[-1]['close'])
            ep   = open_trade['entry']
            pos  = open_trade['position_usdt']
            side = open_trade['side']
            pnl  = (pos * (xp - ep) / ep if side == 'LONG'
                    else pos * (ep - xp) / ep)
            capital += pnl
            trades.append({
                'side': side, 'entry': round(ep, 8), 'exit': round(xp, 8),
                'sl': round(open_trade['sl'], 8), 'tp1': round(open_trade['tp1'], 8),
                'pnl_usdt': round(pnl, 4),
                'pnl_pct':  round(pnl / self.initial_capital * 100, 4),
                'hold': open_trade['hold'],
                'outcome': 'WIN' if pnl > 0 else 'LOSS',
                'exit_reason': 'END_OF_DATA',
            })
            equity_curve.append(capital)

        # ── Metrics ────────────────────────────────────────────────────────────
        n_trades = len(trades)
        wins     = [t for t in trades if t['outcome'] == 'WIN']
        losses   = [t for t in trades if t['outcome'] == 'LOSS']

        win_rate     = len(wins) / n_trades * 100 if n_trades else 0.0
        gross_profit = sum(t['pnl_usdt'] for t in wins)
        gross_loss   = abs(sum(t['pnl_usdt'] for t in losses))
        pf_raw       = gross_profit / gross_loss if gross_loss > 0 else (
                       9999.0 if gross_profit > 0 else 0.0)

        avg_hold = (sum(t['hold'] for t in trades) / n_trades) if n_trades else 0.0

        rr_list = [
            abs(t['tp1'] - t['entry']) / abs(t['entry'] - t['sl'])
            for t in trades
            if abs(t['entry'] - t['sl']) > 1e-12
        ]
        avg_rr = sum(rr_list) / len(rr_list) if rr_list else 0.0

        # Max drawdown
        peak, max_dd = self.initial_capital, 0.0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = (peak - eq) / peak * 100 if peak > 0 else 0.0
            if dd > max_dd:
                max_dd = dd

        # Sharpe (annualised, rf=0, per-candle equity returns)
        sharpe = 0.0
        eq_arr = np.array(equity_curve, dtype=float)
        if len(eq_arr) > 2:
            denom = np.where(eq_arr[:-1] > 0, eq_arr[:-1], 1.0)
            rets  = np.diff(eq_arr) / denom
            std_r = float(np.std(rets))
            if std_r > 1e-12:
                cpy    = self._CPY.get(self.timeframe, 8760)
                sharpe = float(np.mean(rets) / std_r * math.sqrt(cpy))

        total_return = (capital - self.initial_capital) / self.initial_capital * 100

        return {
            'symbol':           self.symbol,
            'timeframe':        self.timeframe,
            'lookback_candles': lookback_candles,
            'initial_capital':  self.initial_capital,
            'final_capital':    round(capital, 2),
            'total_return_pct': round(total_return, 2),
            'n_trades':         n_trades,
            'win_rate':         round(win_rate, 1),
            'profit_factor':    round(min(pf_raw, 9999.0), 3),
            'max_drawdown_pct': round(-max_dd, 2),
            'sharpe_ratio':     round(sharpe, 3),
            'avg_rr':           round(avg_rr, 3),
            'avg_hold_candles': round(avg_hold, 1),
            'gross_profit':     round(gross_profit, 2),
            'gross_loss':       round(gross_loss, 2),
            'equity_curve':     [round(e, 2) for e in equity_curve],
            'trades':           trades,
            'elapsed_sec':      round(time.time() - t0, 1),
        }

    # ── REPORT ─────────────────────────────────────────────────────────────────

    def print_report(self, results: dict) -> None:
        w = 54
        sep = '-' * w
        print(f'\n{sep}')
        print(f"  BACKTEST — {results['symbol']}  {results['timeframe']}")
        print(sep)
        print(f"  Candles       : {results['lookback_candles']}")
        print(f"  Capital       : ${results['initial_capital']:>10,.0f}"
              f"  ->  ${results['final_capital']:>10,.2f}")
        print(f"  Total Return  : {results['total_return_pct']:>+.2f}%")
        print(sep)
        print(f"  Trades        : {results['n_trades']}")
        print(f"  Win Rate      : {results['win_rate']:.1f}%")
        print(f"  Profit Factor : {results['profit_factor']}")
        print(f"  Avg R:R       : 1:{results['avg_rr']:.2f}")
        print(f"  Avg Hold      : {results['avg_hold_candles']:.1f} candles")
        print(sep)
        print(f"  Max Drawdown  : {results['max_drawdown_pct']:.2f}%")
        print(f"  Sharpe Ratio  : {results['sharpe_ratio']:.3f}")
        print(f"  Gross Profit  : ${results['gross_profit']:>10,.2f}")
        print(f"  Gross Loss    : -${results['gross_loss']:>9,.2f}")
        print(sep)
        print(f"  Elapsed       : {results['elapsed_sec']}s\n")

    # ── SAVE ───────────────────────────────────────────────────────────────────

    def save_results(self, results: dict,
                     path: str = 'backtest_results.json') -> None:
        out = {k: v for k, v in results.items() if k != 'trades'}
        out['sample_trades'] = results.get('trades', [])[:20]
        with open(path, 'w') as f:
            json.dump(out, f, indent=2, default=str)
        print(f'[Backtest] Results saved -> {path}')


# ── CLI RUNNER ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import ccxt
    from dotenv import load_dotenv
    load_dotenv()

    exchange = ccxt.binance({
        'options': {'defaultType': 'spot'},
        'enableRateLimit': True,
    })
    feed = DataFeed(exchange)

    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    last_bt = last_results = None

    for sym in symbols:
        print(f"\n{'=' * 50}\nBacktesting {sym}...\n{'=' * 50}")
        bt = Backtester(feed, sym, timeframe='1h',
                        initial_capital=10_000.0, risk_pct=1.5)
        results = bt.run(lookback_candles=500)
        bt.print_report(results)
        last_bt, last_results = bt, results

    if last_bt and last_results:
        last_bt.save_results(last_results)
        print('\nResults saved to backtest_results.json')
