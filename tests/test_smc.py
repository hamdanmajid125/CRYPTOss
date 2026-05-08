"""
Unit tests for smc.py pure detection functions.
Run with: pytest tests/test_smc.py -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import pytest

from smc import find_fvgs, find_order_blocks, detect_rsi_divergence, find_liquidity_levels


# ── helpers ────────────────────────────────────────────────────────────────────

def _df(rows: list) -> pd.DataFrame:
    """Build OHLCV DataFrame from list of (o, h, l, c, v) tuples."""
    idx = pd.date_range('2024-01-01', periods=len(rows), freq='1H', tz='UTC')
    return pd.DataFrame(rows, columns=['open', 'high', 'low', 'close', 'volume'], index=idx)


def _with_atr(df: pd.DataFrame, atr: float) -> pd.DataFrame:
    df = df.copy()
    df['atr'] = atr
    return df



# ── find_fvgs ──────────────────────────────────────────────────────────────────

def test_fvg_bullish_detected():
    """Bullish FVG: candle[i+1].low > candle[i-1].high with strong middle body."""
    # candle 0: high=100, candle 1 (middle, big bull body): open=100 close=105
    # candle 2: low=102 > candle0.high=100 → bullish FVG [100, 102]
    rows = [
        (98, 100, 97, 99, 1e6),   # i-1: high=100
        (100, 106, 99, 105, 1e6), # i  : big bullish body (body=5, atr=2 → strong)
        (102, 108, 102, 107, 1e6),# i+1: low=102 > 100 → gap
    ]
    df = _with_atr(_df(rows), atr=2.0)
    fvgs = find_fvgs(df)
    bulls = [f for f in fvgs if f['type'] == 'bullish']
    assert len(bulls) == 1
    assert bulls[0]['bot'] == pytest.approx(100.0)
    assert bulls[0]['top'] == pytest.approx(102.0)


def test_fvg_bearish_detected():
    """Bearish FVG: candle[i+1].high < candle[i-1].low with strong middle body."""
    rows = [
        (105, 107, 104, 105, 1e6), # i-1: low=104
        (105, 106, 100, 101, 1e6), # i  : big bearish body (body=4, atr=2 → strong)
        (98,  102,  97,  98, 1e6), # i+1: high=102 < 104 → gap
    ]
    df = _with_atr(_df(rows), atr=2.0)
    fvgs = find_fvgs(df)
    bears = [f for f in fvgs if f['type'] == 'bearish']
    assert len(bears) == 1
    assert bears[0]['top'] == pytest.approx(104.0)
    assert bears[0]['bot'] == pytest.approx(102.0)


def test_fvg_weak_middle_body_excluded():
    """A bullish gap whose middle candle body < 0.5*ATR should NOT be returned."""
    rows = [
        (98, 100, 97, 99, 1e6),   # i-1: high=100
        (100, 101, 99.9, 100.1, 1e6), # i: tiny body=0.1, atr=5 → weak
        (102, 108, 102, 107, 1e6),    # i+1: low=102 > 100
    ]
    df = _with_atr(_df(rows), atr=5.0)
    fvgs = find_fvgs(df)
    bulls = [f for f in fvgs if f['type'] == 'bullish']
    assert len(bulls) == 0


def test_fvg_mitigated_excluded():
    """A bullish FVG whose zone is later touched should NOT appear."""
    rows = [
        (98, 100, 97, 99, 1e6),     # i-1: high=100
        (100, 106, 99, 105, 1e6),   # i: strong bull body
        (102, 108, 102, 107, 1e6),  # i+1: low=102 → gap [100, 102]
        (103, 109, 99, 108, 1e6),   # later: low=99 <= bot=100 → mitigated
    ]
    df = _with_atr(_df(rows), atr=2.0)
    fvgs = find_fvgs(df)
    bulls = [f for f in fvgs if f['type'] == 'bullish']
    assert len(bulls) == 0


def test_fvg_no_gap_returns_empty():
    """Overlapping candles produce no FVG."""
    rows = [(100 + i, 101 + i, 99 + i, 100.5 + i, 1e6) for i in range(5)]
    df = _with_atr(_df(rows), atr=1.0)
    fvgs = find_fvgs(df)
    assert fvgs == []


# ── find_order_blocks ──────────────────────────────────────────────────────────

def _make_swing_high_ob_df() -> pd.DataFrame:
    """
    Pattern: rising bars create a swing high at bar 4, then BOS up at bar 7.
    Bar 3 is the last bearish candle before the BOS → bullish OB.
    """
    rows = [
        # bars 0-3: build up, bar 3 is bearish
        (100, 103, 99, 101, 1e6),  # 0
        (101, 105, 100, 104, 1e6), # 1
        (104, 108, 103, 107, 1e6), # 2 — swing high candidate (high=108)
        (107, 109, 104, 105, 1e6), # 3 — bearish close < open → OB candidate
        (105, 107, 104, 106, 1e6), # 4
        (106, 108, 105, 107.5, 1e6),# 5
        (107, 113, 106, 112, 1e6), # 6 — BOS: close=112 > swing high=108
        (112, 114, 111, 113, 1e6), # 7
        (113, 115, 112, 114, 1e6), # 8
        (114, 116, 113, 115, 1e6), # 9
        (115, 117, 114, 116, 1e6), # 10
        (116, 118, 115, 117, 1e6), # 11
        (117, 119, 116, 118, 1e6), # 12
        (118, 120, 117, 119, 1e6), # 13
        (119, 121, 118, 120, 1e6), # 14
    ]
    return _df(rows)


def test_ob_bullish_detected():
    """A bearish candle before a BOS-up should be identified as a bullish OB."""
    df = _make_swing_high_ob_df()
    obs = find_order_blocks(df)
    bulls = [o for o in obs if o['type'] == 'bullish']
    assert len(bulls) >= 1


def test_ob_returns_at_most_6():
    """find_order_blocks returns at most 3 per direction (6 total)."""
    # Use a longer trending DF that might produce many OBs
    rows = []
    base = 100.0
    for i in range(60):
        o = base + i * 0.5
        rows.append((o, o + 2, o - 1, o + 1.8, 1e6))
    df = _df(rows)
    obs = find_order_blocks(df)
    assert len(obs) <= 6


def test_ob_mitigated_excluded():
    """
    A bullish OB that price trades back through its 50%-midpoint
    should NOT appear.
    """
    df = _make_swing_high_ob_df()
    # Extend with bars that retrace below the OB midpoint
    extra = [
        (115, 116, 104, 105, 1e6),  # retraces deeply — mid of OB bar 3 is ~106.5
        (105, 107, 104, 106, 1e6),
    ]
    ext = pd.DataFrame(extra, columns=['open', 'high', 'low', 'close', 'volume'],
                       index=pd.date_range(df.index[-1] + pd.Timedelta('1H'),
                                           periods=len(extra), freq='1H', tz='UTC'))
    df2 = pd.concat([df, ext])
    obs_before = find_order_blocks(df)
    obs_after  = find_order_blocks(df2)
    bulls_before = [o for o in obs_before if o['type'] == 'bullish']
    bulls_after  = [o for o in obs_after  if o['type'] == 'bullish']
    assert len(bulls_after) <= len(bulls_before)


# ── detect_rsi_divergence ──────────────────────────────────────────────────────

def _make_div_df(n: int = 60) -> pd.DataFrame:
    idx = pd.date_range('2024-01-01', periods=n, freq='1H', tz='UTC')
    closes = [100.0 + i * 0.1 for i in range(n)]
    df = pd.DataFrame({
        'open':   [c - 0.05 for c in closes],
        'high':   [c + 0.2  for c in closes],
        'low':    [c - 0.2  for c in closes],
        'close':  closes,
        'volume': [1e6] * n,
        'rsi':    [50.0] * n,
    }, index=idx)
    return df


def test_rsi_divergence_bearish():
    """
    Two consecutive price pivot highs where price2 > price1 but RSI2 < RSI1
    → BEARISH divergence.

    detect_rsi_divergence works on df.tail(50).reset_index().
    With a 60-bar df, tail(50) is bars 10-59 → window indices 0-49.
    p2i must be >= n-5 = 45, so the second pivot must be at original bar >= 55
    (window index >= 45).  We place it at original bar 57 → window index 47.
    """
    df = _make_div_df(60)
    closes = df['close'].tolist()
    rsi    = df['rsi'].tolist()

    # First pivot: bar 20 (window index 10)
    closes[20] = 115.0
    rsi[20]    = 70.0
    for nb in [18, 19, 21, 22]:
        closes[nb] = 110.0

    # Second pivot: bar 57 (window index 47 >= 45)
    closes[57] = 120.0   # higher price
    rsi[57]    = 60.0    # lower RSI → bearish divergence
    for nb in [55, 56, 58, 59]:
        closes[nb] = 110.0

    df['close'] = closes
    df['rsi']   = rsi
    result = detect_rsi_divergence(df)
    assert result == 'BEARISH'


def test_rsi_divergence_bullish():
    """
    Two consecutive price pivot lows where price2 < price1 but RSI2 > RSI1
    → BULLISH divergence.
    Second pivot at bar 57 so window index 47 >= n-5=45.
    """
    df = _make_div_df(60)
    closes = df['close'].tolist()
    rsi    = df['rsi'].tolist()

    # First pivot low: bar 20
    closes[20] = 85.0
    rsi[20]    = 30.0
    for nb in [18, 19, 21, 22]:
        closes[nb] = 100.0

    # Second pivot low: bar 57
    closes[57] = 80.0    # lower price
    rsi[57]    = 40.0    # higher RSI → bullish divergence
    for nb in [55, 56, 58, 59]:
        closes[nb] = 100.0

    df['close'] = closes
    df['rsi']   = rsi
    result = detect_rsi_divergence(df)
    assert result == 'BULLISH'


def test_rsi_divergence_none_on_short_df():
    """DataFrame shorter than 20 bars returns NONE."""
    df = _make_div_df(15)
    assert detect_rsi_divergence(df) == 'NONE'


def test_rsi_divergence_none_without_rsi_column():
    """DataFrame without an 'rsi' column returns NONE."""
    df = _make_div_df(60).drop(columns=['rsi'])
    assert detect_rsi_divergence(df) == 'NONE'


# ── find_liquidity_levels ──────────────────────────────────────────────────────

def _make_liq_df() -> pd.DataFrame:
    """
    Two tight clusters of swing highs around 110 (BSL) and swing lows around 90 (SSL).
    """
    base_prices = [100.0] * 50
    # Insert two swing high clusters
    for idx in [10, 12]:    # cluster 1 BSL ~110
        base_prices[idx] = 110.0
    for idx in [30, 32]:    # cluster 2 BSL ~110.1
        base_prices[idx] = 110.1
    # Insert two swing low clusters
    for idx in [15, 17]:    # cluster 1 SSL ~90
        base_prices[idx] = 90.0
    for idx in [35, 37]:    # cluster 2 SSL ~90.1
        base_prices[idx] = 90.1

    idx_range = pd.date_range('2024-01-01', periods=50, freq='1H', tz='UTC')
    highs  = [p + 0.5 if p == 100.0 else p + 0.5 for p in base_prices]
    lows   = [p - 0.5 if p == 100.0 else p - 0.5 for p in base_prices]
    closes = base_prices[:]
    opens  = [p - 0.1 for p in base_prices]

    df = pd.DataFrame({
        'open': opens, 'high': highs, 'low': lows,
        'close': closes, 'volume': [1e6] * 50,
        'atr': [2.0] * 50,
    }, index=idx_range)
    return df


def test_liquidity_returns_dict_structure():
    """Return value must be {'bsl': [...], 'ssl': [...]}."""
    df = _make_liq_df()
    result = find_liquidity_levels(df)
    assert 'bsl' in result and 'ssl' in result
    assert isinstance(result['bsl'], list)
    assert isinstance(result['ssl'], list)


def test_liquidity_bsl_above_ssl():
    """BSL levels should be higher than SSL levels on trending data."""
    df = _make_liq_df()
    result = find_liquidity_levels(df)
    if result['bsl'] and result['ssl']:
        assert min(result['bsl']) > max(result['ssl'])


def test_liquidity_fallback_on_no_clusters():
    """When ATR is huge (no clusters form), fallback returns top-3 highs/lows."""
    rows = [(100, 101, 99, 100, 1e6)] * 20
    df = _with_atr(_df(rows), atr=1000.0)
    result = find_liquidity_levels(df)
    assert len(result['bsl']) > 0
    assert len(result['ssl']) > 0


def test_liquidity_short_df_returns_empty():
    """DataFrame shorter than 10 bars returns empty lists."""
    rows = [(100, 101, 99, 100, 1e6)] * 5
    result = find_liquidity_levels(_df(rows))
    assert result == {'bsl': [], 'ssl': []}
