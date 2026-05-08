"""
Pure SMC (Smart Money Concepts) detection functions.
Shared by data_feed.py (live) and backtest/engine.py.

All functions accept a pandas DataFrame with OHLCV columns (+ 'atr' when available)
and return plain Python dicts/lists — no exchange, no I/O.
"""
import math
from typing import List, Tuple

import pandas as pd


# ── helpers ────────────────────────────────────────────────────────────────────

def _atr_val(df: pd.DataFrame) -> float:
    """Last non-NaN ATR value from df, or a fallback (recent true-range mean)."""
    if 'atr' in df.columns:
        v = df['atr'].dropna()
        if not v.empty:
            return float(v.iloc[-1])
    # Fallback: mean high-low range of last 14 bars
    hl = (df['high'] - df['low']).tail(14)
    return float(hl.mean()) if not hl.empty else 1.0


def _is_nan(v) -> bool:
    try:
        return math.isnan(float(v))
    except Exception:
        return True


def _swing_highs(arr: list, lookback: int = 2) -> List[Tuple[int, float]]:
    n = len(arr)
    return [(i, arr[i]) for i in range(lookback, n - lookback)
            if arr[i] == max(arr[i - lookback: i + lookback + 1])]


def _swing_lows(arr: list, lookback: int = 2) -> List[Tuple[int, float]]:
    n = len(arr)
    return [(i, arr[i]) for i in range(lookback, n - lookback)
            if arr[i] == min(arr[i - lookback: i + lookback + 1])]


# ── 2.1  find_fvgs ─────────────────────────────────────────────────────────────

def find_fvgs(df: pd.DataFrame) -> list:
    """
    3-candle ICT Fair Value Gap detection.

    Bullish FVG: candle[i+1].low > candle[i-1].high AND middle body >= 0.5*ATR.
    Bearish FVG: candle[i+1].high < candle[i-1].low AND middle body >= 0.5*ATR.

    Mitigation:
      - Bull FVG mitigated when any later candle's low <= zone_bot.
      - Bear FVG mitigated when any later candle's high >= zone_top.

    Returns at most 5 most-recent unmitigated FVGs.
    """
    if len(df) < 3:
        return []

    highs  = df['high'].tolist()
    lows   = df['low'].tolist()
    opens  = df['open'].tolist()
    closes = df['close'].tolist()
    atrs   = df['atr'].tolist() if 'atr' in df.columns else [None] * len(df)
    n = len(df)

    fvgs: list = []

    for i in range(1, n - 1):
        body    = abs(closes[i] - opens[i])
        atr_i   = atrs[i] if (atrs[i] is not None and not _is_nan(atrs[i])) else body * 2
        is_strong = (body >= 0.5 * atr_i) if atr_i > 0 else True

        if not is_strong:
            continue

        # Bullish FVG
        if lows[i + 1] > highs[i - 1]:
            top, bot = float(lows[i + 1]), float(highs[i - 1])
            mitigated = any(lows[j] <= bot for j in range(i + 2, n))
            if not mitigated:
                fvgs.append({'type': 'bullish', 'top': top, 'bot': bot,
                             'mitigated': False, 'bar': i})

        # Bearish FVG
        if highs[i + 1] < lows[i - 1]:
            top, bot = float(lows[i - 1]), float(highs[i + 1])
            mitigated = any(highs[j] >= top for j in range(i + 2, n))
            if not mitigated:
                fvgs.append({'type': 'bearish', 'top': top, 'bot': bot,
                             'mitigated': False, 'bar': i})

    return fvgs[-5:]


# ── 2.2  find_order_blocks ─────────────────────────────────────────────────────

def find_order_blocks(df: pd.DataFrame) -> list:
    """
    Proper ICT Order Block detection using swing-high/low pivots + Break of Structure.

    Bullish OB  = last bearish candle (close < open) before the bar that first closes
                  above the most recent 5-bar swing high (BOS up).
                  Zone = [low, high] of that bearish candle.
                  Mitigated when price later trades back through the OB's 50%-midpoint.

    Bearish OB  = last bullish candle (close > open) before the BOS down.

    Returns at most 3 unmitigated OBs per direction (6 total).
    """
    if len(df) < 15:
        return []

    highs  = df['high'].tolist()
    lows   = df['low'].tolist()
    opens  = df['open'].tolist()
    closes = df['close'].tolist()
    n = len(df)

    sh = _swing_highs(highs, lookback=2)
    sl = _swing_lows(lows,   lookback=2)

    bull_obs: list = []
    bear_obs: list = []

    # Bullish OBs
    for sh_idx, sh_price in sh:
        # Find first BOS close above sh_price
        bos_idx = next((j for j in range(sh_idx + 1, n) if closes[j] > sh_price), None)
        if bos_idx is None:
            continue
        # Last bearish candle in [sh_idx, bos_idx)
        ob_idx = next((m for m in range(bos_idx - 1, sh_idx - 1, -1)
                       if closes[m] < opens[m]), None)
        if ob_idx is None:
            continue
        top = float(highs[ob_idx])
        bot = float(lows[ob_idx])
        mid = (top + bot) / 2
        mitigated = any(lows[j] <= mid for j in range(bos_idx + 1, n))
        if not mitigated:
            bull_obs.append({'price': float(opens[ob_idx]), 'top': top, 'bot': bot,
                             'type': 'bullish', 'mitigated': False, 'bar': ob_idx})

    # Bearish OBs
    for sl_idx, sl_price in sl:
        bos_idx = next((j for j in range(sl_idx + 1, n) if closes[j] < sl_price), None)
        if bos_idx is None:
            continue
        ob_idx = next((m for m in range(bos_idx - 1, sl_idx - 1, -1)
                       if closes[m] > opens[m]), None)
        if ob_idx is None:
            continue
        top = float(highs[ob_idx])
        bot = float(lows[ob_idx])
        mid = (top + bot) / 2
        mitigated = any(highs[j] >= mid for j in range(bos_idx + 1, n))
        if not mitigated:
            bear_obs.append({'price': float(opens[ob_idx]), 'top': top, 'bot': bot,
                             'type': 'bearish', 'mitigated': False, 'bar': ob_idx})

    # 3 most recent per direction
    bull_obs = sorted(bull_obs, key=lambda x: x['bar'])[-3:]
    bear_obs = sorted(bear_obs, key=lambda x: x['bar'])[-3:]
    return bull_obs + bear_obs


# ── 2.3  detect_rsi_divergence ─────────────────────────────────────────────────

def detect_rsi_divergence(df: pd.DataFrame) -> str:
    """
    Pivot-based RSI divergence over the last 50 bars.

    Bearish: two consecutive price pivot highs where price2 > price1 AND RSI2 < RSI1,
             with the second pivot within the last 5 bars.
    Bullish: two consecutive price pivot lows  where price2 < price1 AND RSI2 > RSI1,
             with the second pivot within the last 5 bars.
    """
    if len(df) < 20 or 'rsi' not in df.columns:
        return 'NONE'

    window = df.tail(50).reset_index(drop=True)
    closes = window['close'].tolist()
    rsi    = window['rsi'].tolist()
    n      = len(window)

    ph = _swing_highs(closes, lookback=2)   # price pivot highs
    pl = _swing_lows(closes,  lookback=2)   # price pivot lows

    # Bearish divergence
    if len(ph) >= 2:
        p1i, p1p = ph[-2]
        p2i, p2p = ph[-1]
        if p2i >= n - 5:
            r1 = rsi[p1i] if not _is_nan(rsi[p1i]) else 50.0
            r2 = rsi[p2i] if not _is_nan(rsi[p2i]) else 50.0
            if p2p > p1p and r2 < r1:
                return 'BEARISH'

    # Bullish divergence
    if len(pl) >= 2:
        p1i, p1p = pl[-2]
        p2i, p2p = pl[-1]
        if p2i >= n - 5:
            r1 = rsi[p1i] if not _is_nan(rsi[p1i]) else 50.0
            r2 = rsi[p2i] if not _is_nan(rsi[p2i]) else 50.0
            if p2p < p1p and r2 > r1:
                return 'BULLISH'

    return 'NONE'


# ── 2.4  find_liquidity_levels ─────────────────────────────────────────────────

def find_liquidity_levels(df: pd.DataFrame) -> dict:
    """
    Cluster-based liquidity pool detection over the last 50 bars.

    Algorithm:
      1. Find all swing highs (5-bar pivot) → candidates for BSL.
      2. Cluster highs within 0.3 * ATR of each other.
         A cluster of 2+ swing highs = a BSL pool.
      3. Mark a BSL pool as swept if any later candle's high > level + 0.1*ATR.
         Return only unswept BSL pools.
      4. Same for swing lows → SSL.

    Returns {'bsl': [level, ...], 'ssl': [level, ...]} (plain price lists,
    backward-compatible with claude_agent.py prompt builder).
    """
    if len(df) < 10:
        return {'bsl': [], 'ssl': []}

    window = df.tail(50).reset_index(drop=True)
    highs  = window['high'].tolist()
    lows   = window['low'].tolist()
    n      = len(window)
    atr    = _atr_val(window)
    cluster_th = 0.3 * atr
    sweep_th   = 0.1 * atr

    sh = _swing_highs(highs, lookback=2)
    sl = _swing_lows(lows,   lookback=2)

    def cluster_prices(pivots: list) -> list:
        """Group nearby pivots into clusters with count >= 2."""
        if not pivots:
            return []
        groups: list = []
        cur_prices = [pivots[0][1]]
        cur_indices = [pivots[0][0]]
        for idx, price in pivots[1:]:
            if abs(price - cur_prices[0]) <= cluster_th:
                cur_prices.append(price)
                cur_indices.append(idx)
            else:
                if len(cur_prices) >= 2:
                    groups.append({'level': sum(cur_prices) / len(cur_prices),
                                   'count': len(cur_prices),
                                   'last_bar': max(cur_indices)})
                cur_prices  = [price]
                cur_indices = [idx]
        if len(cur_prices) >= 2:
            groups.append({'level': sum(cur_prices) / len(cur_prices),
                           'count': len(cur_prices),
                           'last_bar': max(cur_indices)})
        return groups

    bsl_pools = cluster_prices(sh)
    ssl_pools = cluster_prices(sl)

    def swept_bsl(pool: dict) -> bool:
        lb = pool['last_bar']
        return any(highs[j] >= pool['level'] + sweep_th for j in range(lb + 1, n))

    def swept_ssl(pool: dict) -> bool:
        lb = pool['last_bar']
        return any(lows[j] <= pool['level'] - sweep_th for j in range(lb + 1, n))

    bsl = sorted([p['level'] for p in bsl_pools if not swept_bsl(p)], reverse=True)
    ssl = sorted([p['level'] for p in ssl_pools if not swept_ssl(p)])

    # Graceful fallback: if no clusters found, return top/bottom N as before
    if not bsl:
        bsl = sorted(window['high'].nlargest(3).tolist(), reverse=True)
    if not ssl:
        ssl = sorted(window['low'].nsmallest(3).tolist())

    return {'bsl': bsl, 'ssl': ssl}
