"""
Phase 4 unit tests — funding rate scoring, OI compression, per-symbol cooldown, ATR% sizing.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from signal_logic import score_confidence, generate_signal
from risk_manager import RiskManager, RiskSettings


# ── shared helpers ─────────────────────────────────────────────────────────────

def _bull_bias(agreement=3):
    return {
        'overall': 'BULLISH', 'agreement': agreement,
        'per_tf': {'15m': 'BULLISH', '1h': 'BULLISH', '4h': 'BULLISH'},
    }

def _bear_bias(agreement=3):
    return {
        'overall': 'BEARISH', 'agreement': agreement,
        'per_tf': {'15m': 'BEARISH', '1h': 'BEARISH', '4h': 'BEARISH'},
    }

def _base_data(overall='BULLISH', agreement=3, fr_rate=0.0, oi_compressed=False):
    bias = _bull_bias(agreement) if overall == 'BULLISH' else _bear_bias(agreement)
    return {
        'price': 100.0, 'atr': 3.0, 'rsi': 50.0,
        'macd': 0.1, 'macd_sig': 0.05,
        'volume_signal': 'NORMAL', 'volume_trend': 'FLAT',
        'fvgs': [], 'order_blocks': [], 'ema200': 95.0,
        'adx': 25.0, 'bb_width_percentile': 50,
        'timeframe_bias': bias,
        'fear_greed': {'value': 50, 'label': 'Neutral'},
        'funding_rate': {'rate': fr_rate, 'label': '', 'bias': 'NEUTRAL'},
        'open_interest': {'oi_usdt': 1e9, 'change_pct': 0.0, 'compressed': oi_compressed},
    }


# ── 4.2a: Funding rate scoring ─────────────────────────────────────────────────

def test_fr_positive_penalises_long():
    """Funding > +0.05% with BULLISH bias → -10 vs neutral FR."""
    base  = score_confidence(_base_data(fr_rate=0.0),  _bull_bias(), {'value': 50})
    penalised = score_confidence(_base_data(fr_rate=0.08), _bull_bias(), {'value': 50})
    assert penalised == base - 10


def test_fr_negative_penalises_short():
    """Funding < -0.05% with BEARISH bias → -10 vs neutral FR."""
    data_neutral = _base_data(overall='BEARISH', fr_rate=0.0)
    data_neutral['macd'] = -0.1; data_neutral['macd_sig'] = -0.05; data_neutral['ema200'] = 105.0
    data_penalised = _base_data(overall='BEARISH', fr_rate=-0.08)
    data_penalised['macd'] = -0.1; data_penalised['macd_sig'] = -0.05; data_penalised['ema200'] = 105.0

    base     = score_confidence(data_neutral,   _bear_bias(), {'value': 50})
    penalised = score_confidence(data_penalised, _bear_bias(), {'value': 50})
    assert penalised == base - 10


def test_fr_extreme_long_boosts_short():
    """Funding > +0.1% with BEARISH bias → +10 (contrarian crowd-extreme confirmation)."""
    data_neutral = _base_data(overall='BEARISH', fr_rate=0.0)
    data_boosted = _base_data(overall='BEARISH', fr_rate=0.12)

    base    = score_confidence(data_neutral, _bear_bias(), {'value': 50})
    boosted = score_confidence(data_boosted, _bear_bias(), {'value': 50})
    assert boosted == base + 10


def test_fr_within_band_no_change():
    """Funding in (-0.05%, +0.05%) → no adjustment."""
    s1 = score_confidence(_base_data(fr_rate=0.03),  _bull_bias(), {'value': 50})
    s2 = score_confidence(_base_data(fr_rate=-0.03), _bull_bias(), {'value': 50})
    s0 = score_confidence(_base_data(fr_rate=0.0),   _bull_bias(), {'value': 50})
    assert s1 == s0
    assert s2 == s0


# ── 4.2b: OI compression scoring ──────────────────────────────────────────────

def test_oi_compressed_reduces_score():
    """OI compressed → -5."""
    normal     = score_confidence(_base_data(oi_compressed=False), _bull_bias(), {'value': 50})
    compressed = score_confidence(_base_data(oi_compressed=True),  _bull_bias(), {'value': 50})
    assert compressed == normal - 5


def test_oi_not_compressed_no_change():
    """OI not compressed → no change."""
    s1 = score_confidence(_base_data(oi_compressed=False), _bull_bias(), {'value': 50})
    s2 = score_confidence(_base_data(oi_compressed=False), _bull_bias(), {'value': 50})
    assert s1 == s2


# ── 4.3: Per-symbol cooldown ───────────────────────────────────────────────────

_STATE_FILE = 'test_risk_phase4.json'

def _make_risk():
    if os.path.exists(_STATE_FILE):
        os.remove(_STATE_FILE)
    return RiskManager(RiskSettings(account_usdt=10000.0), state_file=_STATE_FILE)

def _signal(symbol='BTC/USDT', action='LONG', confidence=80, atr=50.0, entry=30000.0):
    sl = entry - 1.5 * atr
    tp1 = entry + 2.0 * atr
    return {
        'symbol': symbol, 'action': action, 'confidence': confidence,
        'entry': entry, 'sl': sl, 'tp1': tp1, 'rr': '1:3', 'atr': atr,
    }


def test_no_cooldown_initially():
    rm = _make_risk()
    ok, reason = rm.can_trade(_signal())
    assert ok, reason


def test_cooldown_after_winning_close():
    rm = _make_risk()
    sig = _signal(symbol='ETH/USDT')
    rm.record_open('t1', sig)
    rm.record_close('t1', pnl_usdt=50.0)  # win → 60-min cooldown

    ok, reason = rm.can_trade(_signal(symbol='ETH/USDT'))
    assert not ok
    assert 'cooldown' in reason.lower()


def test_cooldown_after_losing_close():
    """Loss → 120-min stop cooldown takes priority over 60-min close cooldown."""
    rm = _make_risk()
    sig = _signal(symbol='SOL/USDT')
    rm.record_open('t2', sig)
    rm.record_close('t2', pnl_usdt=-80.0)  # loss → stop-out

    ok, reason = rm.can_trade(_signal(symbol='SOL/USDT'))
    assert not ok
    assert 'cooldown' in reason.lower()


def test_cooldown_does_not_block_other_symbols():
    """Cooldown on ETH/USDT must not block BTC/USDT."""
    rm = _make_risk()
    sig = _signal(symbol='ETH/USDT')
    rm.record_open('t3', sig)
    rm.record_close('t3', pnl_usdt=-30.0)

    ok, reason = rm.can_trade(_signal(symbol='BTC/USDT'))
    assert ok, reason


def test_cooldown_expires():
    """Simulate cooldown having expired by backdating last_close_time."""
    rm = _make_risk()
    rm.last_close_time['BNB/USDT'] = time.time() - 3700  # 61 min ago → expired
    ok, reason = rm.can_trade(_signal(symbol='BNB/USDT'))
    assert ok, reason


def test_stop_cooldown_still_blocks_after_close_expires():
    """Stop cooldown (120 min) still blocks even when close cooldown (60 min) has expired."""
    rm = _make_risk()
    rm.last_close_time['BNB/USDT'] = time.time() - 3700  # 61 min ago (close expired)
    rm.last_stop_time['BNB/USDT']  = time.time() - 3700  # 61 min ago (stop NOT expired — needs 120 min)

    ok, reason = rm.can_trade(_signal(symbol='BNB/USDT'))
    assert not ok
    assert 'cooldown' in reason.lower()


# ── 4.4: ATR% position sizing ─────────────────────────────────────────────────

def test_atr_normalization_reduces_size_for_volatile_asset():
    """ATR > 2.5% of entry → risk_usdt scaled down."""
    rm = RiskSettings(account_usdt=10000.0, risk_pct=1.0)
    if os.path.exists(_STATE_FILE): os.remove(_STATE_FILE)
    mgr = RiskManager(rm, state_file=_STATE_FILE)

    entry = 100.0
    sl    = 85.0   # 15% sl distance
    atr   = 3.0    # 3% of entry — above 2.5% threshold

    result_no_atr = mgr.position_size(entry, sl, confidence=70, atr=0.0)
    result_with_atr = mgr.position_size(entry, sl, confidence=70, atr=atr)

    assert result_with_atr['risk_usdt'] < result_no_atr['risk_usdt']
    assert result_with_atr['atr_scale'] == pytest.approx(0.025 / 0.03, rel=1e-3)


def test_atr_normalization_no_change_below_threshold():
    """ATR <= 2.5% of entry → no scaling."""
    if os.path.exists(_STATE_FILE): os.remove(_STATE_FILE)
    mgr = RiskManager(RiskSettings(account_usdt=10000.0), state_file=_STATE_FILE)
    entry = 100.0
    sl    = 98.0
    atr   = 2.0  # 2% of entry — below threshold

    result_no_atr   = mgr.position_size(entry, sl, confidence=70, atr=0.0)
    result_with_atr = mgr.position_size(entry, sl, confidence=70, atr=atr)

    assert result_with_atr['risk_usdt'] == result_no_atr['risk_usdt']
    assert result_with_atr['atr_scale'] == pytest.approx(1.0)


def test_atr_normalization_exact_threshold():
    """ATR exactly at 2.5% → no scaling (not strictly above threshold)."""
    if os.path.exists(_STATE_FILE): os.remove(_STATE_FILE)
    mgr = RiskManager(RiskSettings(account_usdt=10000.0), state_file=_STATE_FILE)
    result = mgr.position_size(100.0, 97.0, confidence=70, atr=2.5)
    assert result['atr_scale'] == pytest.approx(1.0)
