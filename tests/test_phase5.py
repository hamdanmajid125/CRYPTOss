"""
Phase 5 unit tests — config loading, structlog signal logging, rolling metrics,
alert callbacks, and .gitignore hygiene.
"""
import sys, os, json, time, tempfile
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from config import Config, SignalConfig, RiskConfig, load_config
from bot_logger import SignalLogger
from risk_manager import RiskManager, RiskSettings


# ── 1. Config loading ──────────────────────────────────────────────────────────

def test_config_defaults():
    cfg = Config()
    assert cfg.signal.sl_mult    == 1.5
    assert cfg.signal.tp1_mult   == 2.0
    assert cfg.signal.tp2_mult   == 3.5
    assert cfg.signal.tp3_mult   == 5.5
    assert cfg.signal.fee_pct    == 0.0005
    assert cfg.signal.min_rr     == 2.0
    assert cfg.risk.atr_pct_threshold  == 0.025
    assert cfg.risk.cooldown_close_sec == 3600
    assert cfg.risk.cooldown_stop_sec  == 7200


def test_config_loads_from_yaml(tmp_path):
    yaml_content = (
        "signal:\n"
        "  sl_mult: 2.0\n"
        "  tp1_mult: 3.0\n"
        "risk:\n"
        "  atr_pct_threshold: 0.03\n"
    )
    cfg_file = tmp_path / 'config.yaml'
    cfg_file.write_text(yaml_content)
    cfg = load_config(str(cfg_file))
    assert cfg.signal.sl_mult == 2.0
    assert cfg.signal.tp1_mult == 3.0
    assert cfg.risk.atr_pct_threshold == 0.03
    # Unspecified keys keep defaults
    assert cfg.signal.fee_pct == 0.0005


def test_config_missing_file_returns_defaults():
    cfg = load_config('/nonexistent/path/config.yaml')
    assert cfg.signal.sl_mult == 1.5


def test_signal_logic_uses_config_multipliers():
    """Changing sl_mult via parameter changes the SL distance."""
    from signal_logic import generate_signal

    def _bull_bias():
        return {'overall': 'BULLISH', 'agreement': 3,
                'per_tf': {'15m': 'BULLISH', '1h': 'BULLISH', '4h': 'BULLISH'}}

    data = {
        'price': 100.0, 'atr': 3.0, 'rsi': 50.0,
        'macd': 0.1, 'macd_sig': 0.05,
        'volume_signal': 'NORMAL', 'volume_trend': 'FLAT',
        'fvgs': [], 'order_blocks': [], 'ema200': 95.0,
        'adx': 25.0, 'bb_width_percentile': 50,
        'timeframe_bias': _bull_bias(),
        'fear_greed': {'value': 50, 'label': 'Neutral'},
    }
    sig_default = generate_signal(data, sl_mult=1.5)
    sig_wide    = generate_signal(data, sl_mult=2.0)

    if sig_default['action'] == 'LONG' and sig_wide['action'] == 'LONG':
        assert sig_wide['sl'] < sig_default['sl']   # wider SL is further from entry


# ── 2. SignalLogger ────────────────────────────────────────────────────────────

def _sample_market_data():
    return {
        'price': 50000.0, 'atr': 800.0, 'rsi': 55.0,
        'adx': 28.0, 'bb_width_percentile': 60,
        'timeframe_bias': {'overall': 'BULLISH', 'agreement': 3},
        'volume_signal': 'HIGH', 'fear_greed': {'value': 45},
        'funding_rate': {'rate': 0.02}, 'open_interest': {'compressed': False},
    }


def test_signal_logger_writes_jsonl(tmp_path):
    path = str(tmp_path / 'signals.jsonl')
    logger = SignalLogger(path=path)
    rule_sig  = {'action': 'LONG',  'confidence': 75}
    final_sig = {'action': 'LONG',  'confidence': 80}
    veto      = {'decision': 'APPROVE', 'confidence_adjustment': 5}

    logger.log_decision('BTC/USDT', _sample_market_data(), rule_sig,
                        veto=veto, final_sig=final_sig)

    assert os.path.exists(path)
    with open(path) as f:
        lines = [l.strip() for l in f if l.strip()]
    assert len(lines) == 1
    rec = json.loads(lines[0])
    assert rec['symbol']          == 'BTC/USDT'
    assert rec['action']          == 'LONG'
    assert rec['confidence']      == 80
    assert rec['rule_confidence'] == 75
    assert rec['veto_decision']   == 'APPROVE'
    assert rec['veto_adj']        == 5
    assert 'features' in rec
    assert rec['features']['price'] == 50000.0


def test_signal_logger_appends_multiple(tmp_path):
    path = str(tmp_path / 'signals.jsonl')
    logger = SignalLogger(path=path)
    for i in range(3):
        logger.log_decision(f'SYM{i}', _sample_market_data(),
                            {'action': 'WAIT', 'confidence': 0})
    with open(path) as f:
        lines = [l for l in f if l.strip()]
    assert len(lines) == 3


def test_signal_logger_read_recent_filters_by_days(tmp_path):
    path = str(tmp_path / 'signals.jsonl')
    logger = SignalLogger(path=path)
    # Write one old entry manually
    old_entry = {'ts': '2020-01-01T00:00:00Z', 'symbol': 'OLD', 'action': 'WAIT'}
    with open(path, 'w') as f:
        f.write(json.dumps(old_entry) + '\n')
    # Write a fresh entry
    logger.log_decision('FRESH', _sample_market_data(), {'action': 'LONG', 'confidence': 70})

    recent = logger.read_recent(days=30)
    symbols = [r['symbol'] for r in recent]
    assert 'FRESH' in symbols
    assert 'OLD' not in symbols


# ── 3. Rolling metrics ────────────────────────────────────────────────────────

def _make_mgr(closes_log):
    return RiskManager(RiskSettings(account_usdt=10000.0),
                       state_file='test_phase5_risk.json',
                       closes_log=closes_log)


def test_rolling_metrics_empty_log(tmp_path):
    mgr = _make_mgr(str(tmp_path / 'closes.jsonl'))
    result = mgr.rolling_metrics(days=30)
    assert result['n_trades'] == 0
    assert result['expectancy'] == 0.0


def test_rolling_metrics_computes_correctly(tmp_path):
    closes_path = str(tmp_path / 'closes.jsonl')
    mgr = _make_mgr(closes_path)

    # Simulate 10 trades: 6 wins (+50), 4 losses (-30)
    now = time.time()
    with open(closes_path, 'w') as f:
        for i in range(6):
            f.write(json.dumps({'ts': now - 3600, 'trade_id': f'w{i}',
                                'symbol': 'BTC/USDT', 'pnl': 50.0}) + '\n')
        for i in range(4):
            f.write(json.dumps({'ts': now - 3600, 'trade_id': f'l{i}',
                                'symbol': 'BTC/USDT', 'pnl': -30.0}) + '\n')

    result = mgr.rolling_metrics(days=30)
    assert result['n_trades']  == 10
    assert result['win_rate']  == pytest.approx(60.0)
    assert result['expectancy'] == pytest.approx((6 * 50 - 4 * 30) / 10)


def test_rolling_metrics_ignores_old_closes(tmp_path):
    closes_path = str(tmp_path / 'closes.jsonl')
    mgr = _make_mgr(closes_path)
    old_ts = time.time() - 35 * 86400  # 35 days ago — outside 30-day window
    with open(closes_path, 'w') as f:
        f.write(json.dumps({'ts': old_ts, 'trade_id': 't1',
                            'symbol': 'ETH/USDT', 'pnl': 100.0}) + '\n')
    result = mgr.rolling_metrics(days=30)
    assert result['n_trades'] == 0


# ── 4. Alert callback ─────────────────────────────────────────────────────────

def test_on_alert_fires_for_bot_pause():
    events = []
    mgr = RiskManager(
        RiskSettings(account_usdt=10000.0, single_loss_pct=1.0,
                     single_loss_pause_sec=3600),
        state_file='test_phase5_risk.json',
        on_alert=lambda ev, data: events.append((ev, data)),
    )
    mgr._check_pause_conditions(-200.0)  # 2% loss → triggers single_loss_pct=1%
    assert any(ev == 'bot_paused' for ev, _ in events)


def test_on_alert_fires_for_drawdown_halt():
    events = []
    mgr = RiskManager(
        RiskSettings(account_usdt=10000.0, drawdown_halt_pct=5.0),
        state_file='test_phase5_risk.json',
        on_alert=lambda ev, data: events.append((ev, data)),
    )
    mgr.peak_balance    = 10000.0
    mgr.current_balance = 9400.0  # 6% drawdown > 5% halt
    mgr.open_trades = []
    sig = {'action': 'LONG', 'confidence': 80, 'symbol': 'BTC/USDT',
           'entry': 30000.0, 'sl': 29000.0, 'tp1': 32000.0, 'rr': '1:3', 'atr': 500.0}
    ok, _ = mgr.can_trade(sig)
    assert not ok
    assert any(ev == 'drawdown_halt' for ev, _ in events)


# ── 5. .gitignore hygiene ─────────────────────────────────────────────────────

def test_gitignore_covers_required_patterns():
    gitignore_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), '.gitignore'
    )
    assert os.path.exists(gitignore_path), '.gitignore must exist'
    content = open(gitignore_path).read()
    for pattern in ['.env', '*.log', '__pycache__/', '*.jsonl']:
        assert pattern in content, f'{pattern!r} missing from .gitignore'
