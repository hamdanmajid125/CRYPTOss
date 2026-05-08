"""
Unit tests for the Phase-1 backtester.
Run with: pytest tests/test_backtester.py -v

Tests:
  1. Bar replay correctness — signal generated at correct bar index
  2. Fee deduction — long trade at SL returns less than breakeven
  3. SL hit on LONG — low crosses SL → trade closed with loss
  4. TP1 hit on LONG — high crosses TP1 → partial close, SL moves to BE
  5. Scale-out math — TP1+TP2+TP3 all hit → full close
  6. Equity curve continuity — monotonic within a flat run
  7. Short SL hit — price rises to SL → loss
"""
import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import pytest

from backtest.engine import (
    BarReplayer, Trade,
    _fill_pnl,
    FEE_PCT, SLIPPAGE_PCT, TP1_FRAC, TP2_FRAC, TP3_FRAC,
)
from signal_logic import generate_signal, score_confidence


# ── Synthetic DataFrame helpers ────────────────────────────────────────────────

def _trending_bull_df(n: int = 300) -> pd.DataFrame:
    """Creates a clear uptrend: each close slightly higher than the last."""
    idx = pd.date_range('2024-01-01', periods=n, freq='1H', tz='UTC')
    closes = [100.0 * (1.0002 ** i) for i in range(n)]
    return pd.DataFrame({
        'open':   [c * 0.999  for c in closes],
        'high':   [c * 1.003  for c in closes],
        'low':    [c * 0.997  for c in closes],
        'close':  closes,
        'volume': [2_000_000.0] * n,
    }, index=idx)


def _make_trade(
    action: str = 'LONG',
    entry: float = 100.0,
    sl: float    = 98.5,
    tp1: float   = 102.0,
    tp2: float   = 103.5,
    tp3: float   = 105.5,
    confidence: int = 75,
    pos_usdt: float = 1000.0,
) -> Trade:
    return Trade(
        symbol='BTC/USDT', action=action, entry=entry,
        sl=sl, tp1=tp1, tp2=tp2, tp3=tp3,
        confidence=confidence, position_usdt=pos_usdt,
        entry_bar=0, entry_time=pd.Timestamp('2024-01-01'),
    )


def _bar(high: float, low: float, close: float = None, ts=None) -> pd.Series:
    if close is None:
        close = (high + low) / 2
    return pd.Series({
        'open': low, 'high': high, 'low': low, 'close': close,
        'volume': 1_000_000.0,
    }, name=ts or pd.Timestamp('2024-01-02'))


# ── Test 1: bar replay correctness ─────────────────────────────────────────────

def test_replay_respects_warmup():
    """No trade should be entered before the warmup index."""
    df = _trending_bull_df(300)
    replayer = BarReplayer(symbol='BTC/USDT', warmup=220, mode='rules')
    _, equity = replayer.run(df)
    initial = equity[0]
    for e in equity[:220]:
        assert e == initial, "Capital changed before warmup ended"


# ── Test 2: fee deduction ──────────────────────────────────────────────────────

def test_fee_deduction_on_breakeven():
    """A trade that exits exactly at entry should show a loss due to fees."""
    entry = 100.0
    exit_price = 100.0   # breakeven price — no price move
    pos_usdt = 1000.0
    pnl = _fill_pnl('LONG', entry, exit_price, 1.0, pos_usdt)
    expected_cost = pos_usdt * 2 * (FEE_PCT + SLIPPAGE_PCT)
    assert pnl == pytest.approx(-expected_cost, rel=1e-6), (
        f"Breakeven trade should lose fees, got {pnl:.4f}")


def test_fee_deduction_profitable_trade():
    """A profitable trade should have fees subtracted from gross profit."""
    entry = 100.0
    tp1   = 102.0
    pos   = 1000.0
    pnl   = _fill_pnl('LONG', entry, tp1, 1.0, pos)
    gross = (tp1 - entry) / entry * pos        # 2 % of 1000 = 20
    cost  = pos * 2 * (FEE_PCT + SLIPPAGE_PCT)
    assert pnl == pytest.approx(gross - cost, rel=1e-6)


# ── Test 3: SL hit on LONG ─────────────────────────────────────────────────────

def test_sl_hit_long():
    """When bar.low crosses SL, the trade should close with a loss."""
    trade = _make_trade('LONG', entry=100.0, sl=98.5)
    bar   = _bar(high=99.0, low=98.0)  # low < sl=98.5

    closed, exit_info = BarReplayer._check_exits(trade, bar)
    assert closed, "SL hit should close the trade"
    assert exit_info['exit_reason'] == 'SL_HIT'
    assert trade.realized_pnl < 0, "SL hit should be a loss"


def test_sl_not_hit_when_low_above_sl():
    """If bar.low > SL, the trade stays open."""
    trade = _make_trade('LONG', entry=100.0, sl=98.5)
    bar   = _bar(high=101.0, low=99.0)  # low > sl=98.5

    closed, _ = BarReplayer._check_exits(trade, bar)
    assert not closed, "SL not hit — trade should stay open"


# ── Test 4: TP1 hit on LONG ────────────────────────────────────────────────────

def test_tp1_hit_long_partial_close():
    """When bar.high crosses TP1, 50 % should be closed and SL moved to entry."""
    trade = _make_trade('LONG', entry=100.0, sl=98.5, tp1=102.0, tp2=103.5, tp3=105.5)
    bar   = _bar(high=102.5, low=100.5)  # high > tp1=102

    closed, _ = BarReplayer._check_exits(trade, bar)
    assert not closed, "TP1 hit alone should NOT fully close the trade"
    assert trade.tp1_hit, "tp1_hit flag should be set"
    assert trade.sl == trade.entry, "SL must move to entry (breakeven) after TP1"
    assert trade.realized_pnl > 0, "Partial TP1 PnL should be positive"


# ── Test 5: scale-out math — all TPs hit ──────────────────────────────────────

def test_all_tps_hit_full_close():
    """If all three TPs are hit across bars, the trade should be fully closed."""
    trade = _make_trade('LONG', entry=100.0, sl=98.5,
                        tp1=102.0, tp2=103.5, tp3=105.5, pos_usdt=1000.0)

    # Bar 1: TP1 hit
    bar1 = _bar(high=102.5, low=100.5)
    closed1, _ = BarReplayer._check_exits(trade, bar1)
    assert not closed1
    assert trade.tp1_hit

    # Bar 2: TP2 hit
    bar2 = _bar(high=104.0, low=101.5)
    closed2, _ = BarReplayer._check_exits(trade, bar2)
    assert not closed2
    assert trade.tp2_hit

    # Bar 3: TP3 hit
    bar3 = _bar(high=106.0, low=104.5)
    closed3, exit3 = BarReplayer._check_exits(trade, bar3)
    assert closed3, "TP3 hit should close the trade"
    assert exit3['exit_reason'] == 'TP3_HIT'

    # Total PnL should equal sum of three partial closes (minus fees)
    pos = 1000.0
    expected = (
        _fill_pnl('LONG', 100.0, 102.0, TP1_FRAC, pos) +
        _fill_pnl('LONG', 100.0, 103.5, TP2_FRAC, pos) +
        _fill_pnl('LONG', 100.0, 105.5, TP3_FRAC, pos)
    )
    assert trade.realized_pnl == pytest.approx(expected, rel=1e-5)


# ── Test 6: equity curve continuity ───────────────────────────────────────────

def test_equity_curve_continuity():
    """Equity curve must have length > 1; capital never negative."""
    # Use trending data — flat price gives NaN RSI (0/0), which dropna removes entirely
    df = _trending_bull_df(500)
    replayer = BarReplayer(symbol='BTC/USDT', warmup=50, mode='rules')
    _, equity = replayer.run(df)
    assert len(equity) > 1, "Equity curve must have more than 1 entry"
    assert all(e >= 0 for e in equity), "Capital should never go negative"


# ── Test 7: SHORT SL hit ───────────────────────────────────────────────────────

def test_sl_hit_short():
    """For a SHORT trade, if bar.high >= SL the trade closes with a loss."""
    trade = _make_trade('SHORT', entry=100.0, sl=101.5,
                        tp1=98.0, tp2=96.5, tp3=94.5)
    bar   = _bar(high=102.0, low=99.0)  # high > sl=101.5

    closed, exit_info = BarReplayer._check_exits(trade, bar)
    assert closed, "SL hit should close SHORT trade"
    assert exit_info['exit_reason'] == 'SL_HIT'
    assert trade.realized_pnl < 0, "SHORT SL hit should be a loss"


# ── Test 8: score_confidence smoke test ───────────────────────────────────────

def test_score_confidence_range():
    """score_confidence must always return a value between 0 and 100."""
    tf_bias = {'overall': 'BULLISH', 'agreement': 2, 'per_tf': {'1h': 'BULLISH', '4h': 'NEUTRAL', '15m': 'BULLISH'}}
    fng     = {'value': 50, 'label': 'Neutral'}
    data    = {'price': 100.0, 'rsi': 55.0, 'macd': 0.1, 'macd_sig': 0.05,
               'volume_signal': 'NORMAL', 'volume_trend': 'FLAT',
               'fvgs': [], 'order_blocks': [], 'ema200': 95.0}
    score = score_confidence(data, tf_bias, fng)
    assert 0 <= score <= 100


# ── Test 9: generate_signal WAIT on MIXED TF ──────────────────────────────────

def test_generate_signal_wait_mixed():
    """MTF MIXED → generate_signal should return WAIT."""
    tf_bias = {'overall': 'MIXED', 'agreement': 1,
               'per_tf': {'1h': 'BULLISH', '4h': 'BEARISH', '15m': 'NEUTRAL'}}
    data = {'price': 100.0, 'atr': 1.0, 'rsi': 50.0, 'macd': 0.0, 'macd_sig': 0.0,
            'volume_signal': 'NORMAL', 'volume_trend': 'FLAT',
            'fvgs': [], 'order_blocks': [], 'ema200': 95.0,
            'adx': 25.0, 'bb_width_percentile': 50,
            'timeframe_bias': tf_bias, 'fear_greed': {'value': 50, 'label': 'Neutral'}}
    sig = generate_signal(data)
    assert sig['action'] == 'WAIT'


# ── Test 10: generate_signal WAIT on low R:R ──────────────────────────────────

def test_generate_signal_wait_very_low_volume():
    """VERY LOW volume should always block a trade, even with a perfect setup."""
    tf_bias = {'overall': 'BULLISH', 'agreement': 3,
               'per_tf': {'1h': 'BULLISH', '4h': 'BULLISH', '15m': 'BULLISH'}}
    data = {'price': 100.0, 'atr': 1.5,
            'rsi': 50.0, 'macd': 0.1, 'macd_sig': 0.05,
            'volume_signal': 'VERY LOW — weak, avoid trading',
            'volume_trend': 'DECREASING', 'fvgs': [], 'order_blocks': [],
            'ema200': 95.0, 'adx': 30.0, 'bb_width_percentile': 60,
            'timeframe_bias': tf_bias, 'fear_greed': {'value': 50, 'label': 'Neutral'}}
    sig = generate_signal(data)
    assert sig['action'] == 'WAIT', "VERY LOW volume must always return WAIT"
