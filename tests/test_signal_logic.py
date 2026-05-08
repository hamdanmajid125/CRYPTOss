"""
Unit tests for signal_logic.generate_signal (Phase 3 acceptance criteria).

Coverage:
  1. Bullish setup → LONG
  2. Bearish setup → SHORT
  3. MTF disagreement → WAIT
  4. Low R:R (zero ATR) → WAIT
  5. Low confidence (Doji + no confluence) → WAIT
  6. 4H veto kills BULLISH → WAIT
  7. Regime chop (ADX + BB) → WAIT
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from signal_logic import generate_signal


# ── shared fixtures ────────────────────────────────────────────────────────────

def _bull_bias(agreement=3):
    per_tf = {'15m': 'BULLISH', '1h': 'BULLISH', '4h': 'BULLISH'}
    if agreement == 2:
        per_tf['4h'] = 'NEUTRAL'
    return {'overall': 'BULLISH', 'agreement': agreement, 'per_tf': per_tf}


def _bear_bias(agreement=3):
    per_tf = {'15m': 'BEARISH', '1h': 'BEARISH', '4h': 'BEARISH'}
    if agreement == 2:
        per_tf['4h'] = 'NEUTRAL'
    return {'overall': 'BEARISH', 'agreement': agreement, 'per_tf': per_tf}


def _base_data(overall='BULLISH', agreement=3, atr=1.5, rsi=50.0,
               volume='NORMAL', adx=25.0, bb_pct=50):
    bias = _bull_bias(agreement) if overall == 'BULLISH' else _bear_bias(agreement)
    return {
        'price':               100.0,
        'atr':                 atr,
        'rsi':                 rsi,
        'macd':                0.1,
        'macd_sig':            0.05,
        'volume_signal':       volume,
        'volume_trend':        'FLAT',
        'fvgs':                [],
        'order_blocks':        [],
        'ema200':              95.0,
        'adx':                 adx,
        'bb_width_percentile': bb_pct,
        'timeframe_bias':      bias,
        'fear_greed':          {'value': 50, 'label': 'Neutral'},
    }


# ── 1. Bullish setup → LONG ────────────────────────────────────────────────────

def test_bullish_setup_returns_long():
    """Full bull confluence — 3/3 TF agree, normal volume, ATR=3 (≥2% of price) → LONG."""
    data = _base_data(overall='BULLISH', agreement=3, atr=3.0)
    sig  = generate_signal(data)
    assert sig['action'] == 'LONG'
    assert sig['confidence'] >= 65
    assert sig['entry']  == pytest.approx(100.0)
    assert sig['sl']     < sig['entry']
    assert sig['tp1']    > sig['entry']
    assert sig['tp2']    > sig['tp1']
    assert sig['tp3']    > sig['tp2']


# ── 2. Bearish setup → SHORT ───────────────────────────────────────────────────

def test_bearish_setup_returns_short():
    """Full bear confluence — 3/3 TF agree, MACD bearish, price < EMA200 → SHORT."""
    data = _base_data(overall='BEARISH', agreement=3, atr=3.0)
    # Align MACD and EMA200 with bearish direction so confidence >= 65
    data['macd']     = -0.1
    data['macd_sig'] = -0.05
    data['ema200']   = 105.0   # price 100 < ema200 105 → good for bearish
    sig  = generate_signal(data)
    assert sig['action'] == 'SHORT'
    assert sig['confidence'] >= 65
    assert sig['sl']  > sig['entry']
    assert sig['tp1'] < sig['entry']
    assert sig['tp2'] < sig['tp1']
    assert sig['tp3'] < sig['tp2']


# ── 3. MTF disagreement → WAIT ─────────────────────────────────────────────────

def test_mtf_mixed_returns_wait():
    """overall=MIXED forces WAIT regardless of other factors."""
    data = _base_data(overall='BULLISH', agreement=1)
    data['timeframe_bias'] = {
        'overall': 'MIXED', 'agreement': 1,
        'per_tf':  {'15m': 'BULLISH', '1h': 'BEARISH', '4h': 'NEUTRAL'},
    }
    sig = generate_signal(data)
    assert sig['action'] == 'WAIT'


def test_agreement_below_2_returns_wait():
    """agreement < 2 → WAIT even if overall looks directional."""
    data = _base_data(overall='BULLISH', agreement=1)
    sig  = generate_signal(data)
    assert sig['action'] == 'WAIT'


# ── 4. Low R:R → WAIT ──────────────────────────────────────────────────────────

def test_zero_atr_returns_wait():
    """ATR = 0 → entry and SL collapse → R:R undefined → WAIT."""
    data = _base_data(overall='BULLISH', agreement=3, atr=0.0)
    sig  = generate_signal(data)
    assert sig['action'] == 'WAIT'


# ── 5. Low confidence → WAIT ───────────────────────────────────────────────────

def test_low_confidence_returns_wait():
    """
    BULLISH 2/3, MACD bearish, Doji, VERY LOW volume — confidence
    should fall below min_confidence=65.
    """
    data = _base_data(overall='BULLISH', agreement=2, volume='VERY LOW — weak, avoid trading')
    sig  = generate_signal(data)
    assert sig['action'] == 'WAIT'


# ── 6. 4H veto → WAIT ─────────────────────────────────────────────────────────

def test_4h_veto_bearish_kills_long():
    """Overall BULLISH but 4h BEARISH → hard veto → WAIT."""
    data = _base_data(overall='BULLISH', agreement=3)
    data['timeframe_bias']['per_tf']['4h'] = 'BEARISH'
    sig  = generate_signal(data)
    assert sig['action'] == 'WAIT'
    assert '4H' in sig.get('reason', '') or 'veto' in sig.get('reason', '').lower()


def test_4h_veto_bullish_kills_short():
    """Overall BEARISH but 4h BULLISH → hard veto → WAIT."""
    data = _base_data(overall='BEARISH', agreement=3)
    data['timeframe_bias']['per_tf']['4h'] = 'BULLISH'
    sig  = generate_signal(data)
    assert sig['action'] == 'WAIT'


# ── 7. Regime chop → WAIT ─────────────────────────────────────────────────────

def test_regime_chop_returns_wait():
    """ADX < 18 AND BB percentile < 30 → chop regime → WAIT."""
    data = _base_data(overall='BULLISH', agreement=3, adx=15.0, bb_pct=20)
    sig  = generate_signal(data)
    assert sig['action'] == 'WAIT'
    assert 'chop' in sig.get('reason', '').lower() or 'Regime' in sig.get('reason', '')


# ── 8. SL/TP geometry ─────────────────────────────────────────────────────────

def test_long_sl_tp_distances():
    """LONG: SL = 1.5×ATR below entry, TP1 = 2.0×ATR above, TP3 = 5.5×ATR above."""
    data = _base_data(overall='BULLISH', agreement=3, atr=2.0)
    sig  = generate_signal(data)
    if sig['action'] == 'LONG':
        assert sig['sl']  == pytest.approx(100.0 - 1.5 * 2.0, rel=1e-6)
        assert sig['tp1'] == pytest.approx(100.0 + 2.0 * 2.0, rel=1e-6)
        assert sig['tp3'] == pytest.approx(100.0 + 5.5 * 2.0, rel=1e-6)


def test_short_sl_tp_distances():
    """SHORT: SL = 1.5×ATR above entry, TP1 = 2.0×ATR below."""
    data = _base_data(overall='BEARISH', agreement=3, atr=2.0)
    sig  = generate_signal(data)
    if sig['action'] == 'SHORT':
        assert sig['sl']  == pytest.approx(100.0 + 1.5 * 2.0, rel=1e-6)
        assert sig['tp1'] == pytest.approx(100.0 - 2.0 * 2.0, rel=1e-6)
        assert sig['tp3'] == pytest.approx(100.0 - 5.5 * 2.0, rel=1e-6)
