"""
Phase-5 Backtesting Engine

- Pulls 6 months of 1h OHLCV from ccxt.binance (public futures feed)
- Calls claude.get_signal() with prompt-hash caching in .cache/signals/
- Simulates Phase-3 trade management: TP1 scale-out -> BE SL -> TP2 -> trail -> TP3
  plus time-stop and invalidation exit
- Outputs backtest_results/: trades.csv, report.md, equity_curve.png
- Ablation: re-runs with each filter disabled to prove each fix's value
"""
import csv
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import ccxt
import numpy as np
import pandas as pd

# ── Constants ──────────────────────────────────────────────────────────────────
CACHE_DIR   = '.cache/signals'
RESULTS_DIR = 'backtest_results'
os.makedirs(CACHE_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

_CPY = {'1m': 525_600, '5m': 105_120, '15m': 35_040,
        '1h': 8_760,   '4h': 2_190,   '1d': 365}

MAX_HOLD_CANDLES = int(os.getenv('MAX_HOURS_IN_TRADE', '8'))


# ── Config ─────────────────────────────────────────────────────────────────────
@dataclass
class BacktestConfig:
    liquidity_tp:  bool = True   # Phase 2: TP1 = nearest BSL/SSL
    regime_filter: bool = True   # Phase 4: skip ADX<18 / BB%ile<25
    time_stop:     bool = True   # Phase 3: close flat trades after MAX_HOLD_CANDLES
    scale_out:     bool = True   # Phase 3: partial close at TP1, move SL to BE


# ── Engine ─────────────────────────────────────────────────────────────────────
class BacktestEngine:

    def __init__(self, feed, agent, symbol: str,
                 timeframe: str = '1h',
                 initial_capital: float = 10_000.0,
                 risk_pct: float = 1.5,
                 cfg: BacktestConfig = None,
                 label: str = 'full'):
        self.feed    = feed
        self.agent   = agent
        self.symbol  = symbol
        self.tf      = timeframe
        self.initial = float(initial_capital)
        self.rpct    = float(risk_pct)
        self.cfg     = cfg or BacktestConfig()
        self.label   = label

    # ── DATA FETCH ─────────────────────────────────────────────────────────────

    def fetch_data(self, months: int = 6) -> pd.DataFrame:
        ex    = ccxt.binance({'enableRateLimit': True,
                              'options': {'defaultType': 'future'}})
        since = int((time.time() - months * 30 * 86400) * 1000)
        bars  = []
        while True:
            chunk = ex.fetch_ohlcv(self.symbol, self.tf, since=since, limit=1000)
            if not chunk:
                break
            bars.extend(chunk)
            if len(chunk) < 1000:
                break
            since = chunk[-1][0] + 1
            time.sleep(ex.rateLimit / 1000)

        df = pd.DataFrame(bars, columns=['ts', 'open', 'high', 'low', 'close', 'volume'])
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('ts', inplace=True)
        for col in ('open', 'high', 'low', 'close', 'volume'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        print(f'[Backtest] {self.symbol} — {len(df)} candles ({months}m)')
        return df

    # ── BUILD MARKET DATA FROM HISTORICAL SLICE ───────────────────────────────

    def _build_data(self, hist: pd.DataFrame) -> Optional[dict]:
        last = hist.iloc[-1]
        required = ('ema20', 'ema50', 'rsi', 'atr', 'macd', 'macd_sig',
                    'bb_upper', 'bb_lower', 'volume_sma')
        for c in required:
            if c not in hist.columns or pd.isna(last[c]):
                return None

        price  = float(last['close'])
        prev24 = hist.iloc[-24] if len(hist) >= 24 else hist.iloc[0]
        ch24   = (price - float(prev24['close'])) / float(prev24['close']) * 100

        # Regime metrics
        win      = hist.tail(100)
        cur_bbw  = float(last['bb_upper'] - last['bb_lower'])
        cur_atr  = float(last['atr'])
        bbw_s    = (win['bb_upper'] - win['bb_lower']).dropna()
        atr_s    = win['atr'].dropna()
        bbp = int(round((bbw_s < cur_bbw).sum() / len(bbw_s) * 100)) if len(bbw_s) > 1 else 50
        atp = int(round((atr_s  < cur_atr).sum() / len(atr_s)  * 100)) if len(atr_s)  > 1 else 50
        adx_v = float(last['adx']) if 'adx' in last.index and not pd.isna(last.get('adx', float('nan'))) else 0.0

        pat = hist.tail(100)
        tf_bias = self._bias(hist)

        return {
            'symbol':              self.symbol,
            'timeframe':           self.tf,
            'timestamp':           str(last.name),
            'price':               price,
            'change24h':           round(ch24, 2),
            'rsi':                 round(float(last['rsi']), 1),
            'macd':                round(float(last['macd']), 8),
            'macd_sig':            round(float(last['macd_sig']), 8),
            'atr':                 round(cur_atr, 8),
            'ema20':               round(float(last['ema20']), 8),
            'ema50':               round(float(last['ema50']), 8),
            'ema200':              (float(last['ema200'])
                                    if 'ema200' in last.index and not pd.isna(last.get('ema200', float('nan')))
                                    else None),
            'bb_upper':            round(float(last['bb_upper']), 8),
            'bb_lower':            round(float(last['bb_lower']), 8),
            'volume':              float(last['volume']),
            'volume_sma':          float(last['volume_sma']) if not pd.isna(last['volume_sma']) else 0,
            'volume_signal':       self._vol_signal(hist),
            'volume_trend':        self._vol_trend(hist),
            'rsi_divergence':      self.feed.detect_rsi_divergence(hist),
            'candle_pattern':      self.feed.detect_candle_pattern(hist),
            'news_sentiment':      'NEUTRAL',
            'timeframe_bias':      tf_bias,
            'fear_greed':          {'value': 50, 'label': 'Neutral'},
            'liquidity':           self.feed.find_liquidity_levels(pat),
            'order_blocks':        self.feed.find_order_blocks(pat),
            'fvgs':                self.feed.find_fvgs(pat),
            'bos_choch':           self.feed.detect_bos_choch(hist),
            'regime':              self.feed.detect_regime(hist),
            'funding_rate':        {'rate': 0.0, 'label': 'Neutral', 'bias': 'NEUTRAL'},
            'structure':           self.feed.market_structure(hist),
            'adx':                 round(adx_v, 2),
            'bb_width_percentile': bbp,
            'atr_percentile':      atp,
        }

    def _bias(self, hist: pd.DataFrame) -> dict:
        last  = hist.iloc[-1]
        close = float(last['close'])
        ema20 = float(last['ema20']) if not pd.isna(last.get('ema20', float('nan'))) else close
        ema50 = float(last['ema50']) if not pd.isna(last.get('ema50', float('nan'))) else close
        lb    = min(5, len(hist) - 1)
        prev  = float(hist.iloc[-(lb + 1)]['ema20']) if lb > 0 else ema20
        up    = ema20 > prev
        if close > ema20 > ema50 and up:
            overall, agr = 'BULLISH', 2
        elif close < ema20 < ema50 and not up:
            overall, agr = 'BEARISH', 2
        else:
            overall, agr = 'MIXED', 1
        return {'overall': overall, 'agreement': agr, 'per_tf': {self.tf: overall}}

    def _vol_signal(self, hist: pd.DataFrame) -> str:
        last = hist.iloc[-1]
        vsma = last.get('volume_sma')
        if vsma is None or pd.isna(vsma) or vsma == 0:
            return 'NORMAL'
        r = float(last['volume']) / float(vsma)
        if r >= 2.0: return 'VERY HIGH -- strong confirmation'
        if r >= 1.5: return 'HIGH -- confirms direction'
        if r <= 0.5: return 'VERY LOW -- weak, avoid trading'
        if r <= 0.7: return 'LOW -- signal is weaker'
        return 'NORMAL'

    def _vol_trend(self, hist: pd.DataFrame) -> str:
        v = hist['volume'].tail(5).values
        if len(v) < 5:          return 'FLAT'
        if v[-1] > v[0] * 1.3: return 'INCREASING -- momentum building'
        if v[-1] < v[0] * 0.7: return 'DECREASING -- momentum fading'
        return 'FLAT'

    # ── CACHED CLAUDE SIGNAL ───────────────────────────────────────────────────

    def _get_signal(self, data: dict) -> Optional[dict]:
        if self.cfg.regime_filter:
            adx = data.get('adx', 0)
            bbp = data.get('bb_width_percentile', 50)
            if adx < 18 or bbp < 25:
                return {'action': 'WAIT', 'confidence': 0,
                        'reason': f'REGIME_SKIP ADX={adx:.1f} BB%={bbp}'}

        try:
            prompt = self.agent.build_prompt(data)
        except Exception:
            return None

        h          = hashlib.md5(prompt.encode('utf-8')).hexdigest()
        cache_path = os.path.join(CACHE_DIR, f'{h}.json')

        if os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    return json.load(f)
            except Exception:
                pass

        try:
            sig = self.agent.get_signal(data)
            if sig:
                with open(cache_path, 'w') as f:
                    json.dump(sig, f)
            return sig
        except Exception as e:
            print(f'[Backtest] Claude error: {e}')
            return None

    # ── LIQUIDITY-ANCHORED TP ─────────────────────────────────────────────────

    def _liq_tp1(self, data: dict, side: str, entry: float, atr: float) -> Optional[float]:
        if not self.cfg.liquidity_tp:
            return entry + atr * 2.0 if side == 'LONG' else entry - atr * 2.0
        liq  = data.get('liquidity', {'bsl': [], 'ssl': []})
        pool = liq.get('bsl', []) if side == 'LONG' else liq.get('ssl', [])
        lo, hi = atr * 0.8, atr * 2.5
        best: Optional[Tuple[float, float]] = None
        for lvl in pool:
            dist = (lvl - entry) if side == 'LONG' else (entry - lvl)
            if lo <= dist <= hi:
                if best is None or dist < best[0]:
                    best = (dist, lvl)
        return best[1] if best else None

    # ── MAIN RUN ──────────────────────────────────────────────────────────────

    def run(self, df: pd.DataFrame, warmup: int = 60) -> dict:
        t0       = time.time()
        n        = len(df)
        capital  = self.initial
        equity: List[float] = [capital]
        trades: List[dict]  = []
        trade: Optional[dict] = None
        api_calls = cache_hits = 0

        for i in range(warmup, n - 1):
            hist = df.iloc[:i + 1]
            cur  = hist.iloc[-1]

            # ── Advance open trade ───────────────────────────────────────────
            if trade is not None:
                done, capital, record = self._step(trade, cur, capital)
                if done:
                    trades.append(record)
                    trade = None

            equity.append(capital)

            if trade is not None:
                continue   # still in a trade — no new signal

            # ── Build data and fetch signal ──────────────────────────────────
            data = self._build_data(hist)
            if data is None:
                continue

            # Cache tracking (check before calling, since _get_signal may cache)
            try:
                h = hashlib.md5(self.agent.build_prompt(data).encode()).hexdigest()
                if os.path.exists(os.path.join(CACHE_DIR, f'{h}.json')):
                    cache_hits += 1
                else:
                    api_calls += 1
            except Exception:
                api_calls += 1

            sig = self._get_signal(data)
            if not sig or sig.get('action') not in ('LONG', 'SHORT'):
                continue

            action  = sig['action']
            entry_p = float(df.iloc[i + 1]['open'])   # enter at next candle open (no lookahead)
            atr_v   = float(cur['atr'])
            if entry_p <= 0 or atr_v <= 0:
                continue

            sl_p = entry_p - atr_v * 1.5 if action == 'LONG' else entry_p + atr_v * 1.5
            tp1  = self._liq_tp1(data, action, entry_p, atr_v)
            if tp1 is None:
                continue   # no liquidity level in range — skip per Phase 2

            tp2 = entry_p + atr_v * 3.0 if action == 'LONG' else entry_p - atr_v * 3.0
            tp3 = entry_p + atr_v * 4.5 if action == 'LONG' else entry_p - atr_v * 4.5

            try:
                inv = float(sig.get('invalidation', 0) or 0)
            except (TypeError, ValueError):
                inv = 0.0

            sl_pct = abs(entry_p - sl_p) / entry_p
            if sl_pct <= 0:
                continue

            risk_usd  = capital * self.rpct / 100
            usd_total = risk_usd / sl_pct   # notional position size

            trade = {
                'side':        action,
                'entry':       entry_p,
                'sl':          sl_p,
                'sl_orig':     sl_p,
                'tp1':         tp1,
                'tp1_orig':    tp1,
                'tp2':         tp2,
                'tp3':         tp3,
                'usd':         usd_total,  # remaining notional
                'usd_orig':    usd_total,
                'atr':         atr_v,
                'inv':         inv,
                'stage':       1,
                'age':         0,
                'pnl_partial': 0.0,
                'confidence':  sig.get('confidence', 0),
                'setup_type':  sig.get('setup_type', ''),
                'tp1_anchor':  sig.get('tp1_anchor', ''),
            }

        # Force-close any trade still open
        if trade is not None:
            xp  = float(df.iloc[-1]['close'])
            pnl = self._pnl(trade, xp) + trade['pnl_partial']
            capital += pnl
            trades.append(self._record(trade, xp, pnl, 'END_OF_DATA'))

        return self._metrics(trades, equity, capital, t0, api_calls, cache_hits)

    # ── STATE MACHINE (per candle) ────────────────────────────────────────────

    def _step(self, t: dict, cur, capital: float) -> Tuple[bool, float, dict]:
        hi   = float(cur['high'])
        lo   = float(cur['low'])
        cls  = float(cur['close'])
        is_l = t['side'] == 'LONG'

        # SL hit — all stages
        sl_hit = (lo <= t['sl']) if is_l else (hi >= t['sl'])
        if sl_hit:
            pnl = self._pnl(t, t['sl']) + t['pnl_partial']
            capital += pnl
            return True, capital, self._record(t, t['sl'], pnl, 'SL_HIT')

        if t['stage'] == 1:
            t['age'] += 1

            # Invalidation: candle CLOSES past the level
            if t['inv'] > 0:
                inv_hit = (cls <= t['inv']) if is_l else (cls >= t['inv'])
                if inv_hit:
                    pnl = self._pnl(t, cls) + t['pnl_partial']
                    capital += pnl
                    return True, capital, self._record(t, cls, pnl, 'INVALIDATION')

            # Time-stop: age exceeded AND trade is flat (-0.3R to +0.3R)
            if self.cfg.time_stop and t['age'] >= MAX_HOLD_CANDLES:
                r_dist = abs(t['entry'] - t['sl_orig'])
                pnl_cur = self._pnl(t, cls)
                if r_dist > 0 and -0.3 * r_dist <= pnl_cur <= 0.3 * r_dist:
                    total = pnl_cur + t['pnl_partial']
                    capital += total
                    return True, capital, self._record(t, cls, total, 'TIME_STOP')

            # TP1 hit
            tp1_hit = (hi >= t['tp1']) if is_l else (lo <= t['tp1'])
            if tp1_hit:
                if self.cfg.scale_out:
                    # Close 50%, move SL to breakeven+buffer
                    close_usd = t['usd'] * 0.50
                    partial   = self._pnl_usd(t, t['tp1'], close_usd)
                    t['pnl_partial'] += partial
                    capital          += partial
                    t['usd']         -= close_usd
                    be = round(t['entry'] * 1.001 if is_l else t['entry'] * 0.999, 8)
                    t['sl']   = be
                    t['stage'] = 2
                else:
                    pnl = self._pnl(t, t['tp1']) + t['pnl_partial']
                    capital += pnl
                    return True, capital, self._record(t, t['tp1'], pnl, 'TP1_HIT')

        elif t['stage'] == 2:
            tp2_hit = (hi >= t['tp2']) if is_l else (lo <= t['tp2'])
            if tp2_hit:
                close_usd = t['usd'] * 0.60   # 30% of original
                partial   = self._pnl_usd(t, t['tp2'], close_usd)
                t['pnl_partial'] += partial
                capital          += partial
                t['usd']         -= close_usd
                t['sl']   = t['tp1_orig']   # SL -> TP1 to lock profit
                t['stage'] = 3

        elif t['stage'] == 3:
            tp3_hit = (hi >= t['tp3']) if is_l else (lo <= t['tp3'])
            if tp3_hit:
                pnl = self._pnl(t, t['tp3']) + t['pnl_partial']
                capital += pnl
                return True, capital, self._record(t, t['tp3'], pnl, 'TP3_HIT')
            # Trail SL
            atr = t['atr']
            if atr > 0:
                if is_l:
                    new_sl = round(max(t['entry'] + 1.5 * atr, lo - 0.5 * atr), 8)
                    if new_sl > t['sl']:
                        t['sl'] = new_sl
                else:
                    new_sl = round(min(t['entry'] - 1.5 * atr, hi + 0.5 * atr), 8)
                    if new_sl < t['sl']:
                        t['sl'] = new_sl

        return False, capital, {}

    # ── PnL HELPERS ───────────────────────────────────────────────────────────

    def _pnl(self, t: dict, exit_p: float) -> float:
        """PnL for current remaining notional."""
        e = t['entry']
        if t['side'] == 'LONG':
            return t['usd'] * (exit_p - e) / e
        return t['usd'] * (e - exit_p) / e

    def _pnl_usd(self, t: dict, exit_p: float, close_usd: float) -> float:
        e = t['entry']
        if t['side'] == 'LONG':
            return close_usd * (exit_p - e) / e
        return close_usd * (e - exit_p) / e

    def _record(self, t: dict, exit_p: float, total_pnl: float, reason: str) -> dict:
        return {
            'side':          t['side'],
            'entry':         round(t['entry'], 8),
            'exit':          round(exit_p, 8),
            'sl_orig':       round(t['sl_orig'], 8),
            'tp1':           round(t['tp1_orig'], 8),
            'tp2':           round(t['tp2'], 8),
            'tp3':           round(t['tp3'], 8),
            'stage_reached': t['stage'],
            'pnl_usdt':      round(total_pnl, 4),
            'outcome':       'WIN' if total_pnl > 0 else 'LOSS',
            'exit_reason':   reason,
            'confidence':    t.get('confidence', 0),
            'setup_type':    t.get('setup_type', ''),
            'tp1_anchor':    t.get('tp1_anchor', ''),
        }

    # ── METRICS ───────────────────────────────────────────────────────────────

    def _metrics(self, trades, equity, capital, t0, api_calls, cache_hits) -> dict:
        n     = len(trades)
        wins  = [t for t in trades if t['outcome'] == 'WIN']
        loss  = [t for t in trades if t['outcome'] == 'LOSS']
        gp    = sum(t['pnl_usdt'] for t in wins)
        gl    = abs(sum(t['pnl_usdt'] for t in loss))
        wr    = len(wins) / n * 100 if n else 0.0
        pf    = gp / gl if gl > 0 else (9999.0 if gp > 0 else 0.0)

        peak, mdd = self.initial, 0.0
        for eq in equity:
            if eq > peak: peak = eq
            dd = (peak - eq) / peak * 100 if peak > 0 else 0.0
            if dd > mdd: mdd = dd

        sharpe = 0.0
        arr    = np.array(equity, dtype=float)
        if len(arr) > 2:
            denom  = np.where(arr[:-1] > 0, arr[:-1], 1.0)
            rets   = np.diff(arr) / denom
            std_r  = float(np.std(rets))
            if std_r > 1e-12:
                sharpe = float(np.mean(rets) / std_r * math.sqrt(_CPY.get(self.tf, 8760)))

        return {
            'label':            self.label,
            'symbol':           self.symbol,
            'timeframe':        self.tf,
            'initial_capital':  self.initial,
            'final_capital':    round(capital, 2),
            'total_return_pct': round((capital - self.initial) / self.initial * 100, 2),
            'n_trades':         n,
            'win_rate':         round(wr, 1),
            'profit_factor':    round(min(pf, 9999.0), 3),
            'max_drawdown_pct': round(mdd, 2),
            'sharpe_ratio':     round(sharpe, 3),
            'gross_profit':     round(gp, 2),
            'gross_loss':       round(gl, 2),
            'equity_curve':     [round(e, 2) for e in equity],
            'trades':           trades,
            'elapsed_sec':      round(time.time() - t0, 1),
            'api_calls':        api_calls,
            'cache_hits':       cache_hits,
            'cfg': {
                'liquidity_tp':  self.cfg.liquidity_tp,
                'regime_filter': self.cfg.regime_filter,
                'time_stop':     self.cfg.time_stop,
                'scale_out':     self.cfg.scale_out,
            },
        }


# ── OUTPUT WRITERS ─────────────────────────────────────────────────────────────

def save_trades_csv(trades: list, path: str):
    if not trades:
        print(f'[Backtest] No trades — skipping {path}')
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=trades[0].keys())
        w.writeheader()
        w.writerows(trades)
    print(f'[Backtest] {len(trades)} trades -> {path}')


def save_report_md(results: dict, path: str, ablation_rows: list = None):
    r = results
    lines = [
        f'# Backtest Report — {r.get("symbol","ALL")} {r.get("timeframe","")}',
        '',
        '## Summary',
        '| Metric | Value |',
        '|--------|-------|',
        f'| Initial Capital | ${r.get("initial_capital", 0):,.0f} |',
        f'| Final Capital | ${r.get("final_capital", 0):,.2f} |',
        f'| Total Return | {r.get("total_return_pct", 0):+.2f}% |',
        f'| Trades | {r.get("n_trades", 0)} |',
        f'| Win Rate | {r.get("win_rate", 0):.1f}% |',
        f'| Profit Factor | {r.get("profit_factor", 0)} |',
        f'| Max Drawdown | {r.get("max_drawdown_pct", 0):.2f}% |',
        f'| Sharpe Ratio | {r.get("sharpe_ratio", 0):.3f} |',
        f'| Gross Profit | ${r.get("gross_profit", 0):,.2f} |',
        f'| Gross Loss | ${r.get("gross_loss", 0):,.2f} |',
        f'| API calls | {r.get("api_calls", "N/A")} |',
        f'| Cache hits | {r.get("cache_hits", "N/A")} |',
        '',
    ]
    if ablation_rows:
        lines += [
            '## Ablation Report',
            '',
            '| Config | Trades | Win% | PF | Sharpe | MaxDD% | Return% |',
            '|--------|--------|------|----|--------|--------|---------|',
        ]
        for ar in ablation_rows:
            lines.append(
                f'| {ar["label"]} | {ar["n_trades"]} | {ar["win_rate"]:.1f}% '
                f'| {ar["profit_factor"]} | {ar["sharpe_ratio"]:.3f} '
                f'| {ar["max_drawdown_pct"]:.2f}% | {ar["total_return_pct"]:+.2f}% |'
            )
        lines.append('')
        lines.append('**Key:** `no_liq_tp` = ATR TPs instead of BSL/SSL | '
                     '`no_regime` = always run LLM | '
                     '`no_time_stop` = no time expiry | '
                     '`no_scale_out` = close 100% at TP1')
    with open(path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f'[Backtest] Report -> {path}')


def save_equity_png(equity: list, path: str, label: str = ''):
    try:
        import matplotlib  # type: ignore[import]
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt  # type: ignore[import]
        _, ax = plt.subplots(figsize=(14, 5))
        ax.plot(equity, linewidth=1.2, color='#00d4aa')
        ax.axhline(equity[0], color='gray', linewidth=0.7, linestyle='--', label='Start')
        ax.fill_between(range(len(equity)), equity[0], equity,
                        alpha=0.12, color='#00d4aa')
        ax.set_title(f'Equity Curve — {label}', fontsize=13)
        ax.set_xlabel('Candle')
        ax.set_ylabel('Capital (USDT)')
        ax.grid(True, alpha=0.25)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()
        print(f'[Backtest] Equity PNG -> {path}')
    except Exception as e:
        print(f'[Backtest] Equity PNG skipped: {e}')


# ── ABLATION RUNNER ────────────────────────────────────────────────────────────

def run_ablation(feed, agent, symbol: str, df: pd.DataFrame,
                 initial_capital: float = 10_000.0,
                 risk_pct: float = 1.5,
                 timeframe: str = '1h') -> list:
    configs = [
        ('full',         BacktestConfig()),
        ('no_liq_tp',    BacktestConfig(liquidity_tp=False)),
        ('no_regime',    BacktestConfig(regime_filter=False)),
        ('no_time_stop', BacktestConfig(time_stop=False)),
        ('no_scale_out', BacktestConfig(scale_out=False)),
    ]
    results = []
    for lbl, cfg in configs:
        print(f'[Ablation] {lbl}...')
        eng = BacktestEngine(feed, agent, symbol, timeframe,
                             initial_capital, risk_pct, cfg, lbl)
        res = eng.run(df)
        results.append(res)
    return results


# ── CLI RUNNER ─────────────────────────────────────────────────────────────────

def main():
    from dotenv import load_dotenv
    load_dotenv()

    import ccxt as _ccxt
    from data_feed import DataFeed
    from claude_agent import ClaudeAgent

    ex    = _ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})
    feed  = DataFeed(ex, cryptopanic_token='')
    agent = ClaudeAgent(api_key=os.getenv('ANTHROPIC_API_KEY', ''))

    initial_capital = float(os.getenv('ACCOUNT_USDT', '10000'))
    risk_pct        = float(os.getenv('RISK_PCT', '1.5'))

    symbols = [
        ('BTC/USDT', '1h'),
        ('ETH/USDT', '1h'),
        ('SOL/USDT', '1h'),
        ('BNB/USDT', '1h'),
    ]

    all_ablation: list = []

    for sym, tf in symbols:
        print(f'\n{"="*60}')
        print(f'  {sym}  {tf}')
        print(f'{"="*60}')
        slug = sym.replace('/', '-')

        # Fetch + pre-compute indicators once for all ablation runs
        eng_base = BacktestEngine(feed, agent, sym, tf, initial_capital, risk_pct)
        df_raw   = eng_base.fetch_data(months=6)
        df       = feed.add_indicators(df_raw)
        print(f'[Backtest] {len(df)} candles after indicator warmup')

        ablation = run_ablation(feed, agent, sym, df, initial_capital, risk_pct, tf)
        full_res  = ablation[0]
        all_ablation.extend(ablation)

        save_trades_csv(full_res['trades'],
                        os.path.join(RESULTS_DIR, f'{slug}_trades.csv'))
        save_equity_png(full_res['equity_curve'],
                        os.path.join(RESULTS_DIR, f'{slug}_equity.png'),
                        f'{sym} {tf}')
        save_report_md(full_res,
                       os.path.join(RESULTS_DIR, f'{slug}_report.md'),
                       ablation_rows=ablation)

    # Global ablation summary
    save_report_md(
        {'symbol': 'ALL', 'timeframe': '1h',
         'initial_capital': initial_capital,
         'final_capital': 0, 'total_return_pct': 0,
         'n_trades': sum(r['n_trades'] for r in all_ablation if r['label'] == 'full'),
         'win_rate': 0, 'profit_factor': 0, 'max_drawdown_pct': 0,
         'sharpe_ratio': 0, 'gross_profit': 0, 'gross_loss': 0,
         'api_calls': 'see per-symbol', 'cache_hits': 'see per-symbol'},
        os.path.join(RESULTS_DIR, 'ablation_summary.md'),
        ablation_rows=all_ablation,
    )
    print('\n[Backtest] All done. Results in backtest_results/')


# Keep the original Backtester class as a lightweight alias for the /backtest API route
class Backtester(BacktestEngine):
    """Thin wrapper — keeps the /backtest/{symbol} FastAPI route working."""
    def __init__(self, feed, symbol: str, timeframe: str = '1h',
                 initial_capital: float = 10_000.0, risk_pct: float = 1.5):
        from claude_agent import ClaudeAgent
        agent = ClaudeAgent(api_key=os.getenv('ANTHROPIC_API_KEY', 'backtest'))
        super().__init__(feed, agent, symbol, timeframe, initial_capital, risk_pct)

    def run(self, lookback_candles: int = 500) -> dict:  # type: ignore[override]
        df = self.feed.get_ohlcv(self.symbol, self.tf, limit=lookback_candles)
        df = self.feed.add_indicators(df)
        return super().run(df)


if __name__ == '__main__':
    main()
