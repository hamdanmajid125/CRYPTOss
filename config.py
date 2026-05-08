"""
Config loader — reads config.yaml, exposes typed Pydantic models.
Falls back to defaults if config.yaml is absent.

Usage:
    from config import cfg
    print(cfg.signal.sl_mult)
"""
from __future__ import annotations

import os
from pydantic import BaseModel


class SignalConfig(BaseModel):
    sl_mult:            float = 1.5
    tp1_mult:           float = 2.0
    tp2_mult:           float = 3.5
    tp3_mult:           float = 5.5
    fee_pct:            float = 0.0005
    min_rr:             float = 2.0
    min_confidence:     int   = 65
    adx_chop_threshold:  float = 25.0
    bb_pct_chop:         float = 30.0
    ema200_hard_gate:    bool  = True
    session_filter:      bool  = True
    session_start_utc:   int   = 8
    session_end_utc:     int   = 20
    confluence_required: bool  = True


class RiskConfig(BaseModel):
    account_usdt:          float = 10000.0
    risk_pct:              float = 1.0
    max_trade_usdt:        float = 150.0
    min_confidence:        int   = 72
    daily_loss_limit:      float = 250.0
    min_rr:                float = 3.0
    max_concurrent:        int   = 3
    atr_pct_threshold:     float = 0.025
    cooldown_close_sec:    int   = 3600
    cooldown_stop_sec:     int   = 7200
    drawdown_halt_pct:     float = 10.0
    drawdown_reduce_pct:   float = 5.0
    consec_3_pause_sec:    int   = 7200
    consec_5_pause_sec:    int   = 259200
    single_loss_pct:       float = 3.0
    single_loss_pause_sec: int   = 3600
    tp1_atr_min:           float = 0.5
    tp1_atr_max:           float = 3.0


class SmcConfig(BaseModel):
    fvg_max:              int   = 5
    ob_max:               int   = 6
    rsi_lookback:         int   = 50
    liq_lookback:         int   = 50
    liq_cluster_atr_mult: float = 0.3
    liq_min_cluster:      int   = 2
    liq_swept_atr_mult:   float = 0.1
    pivot_lookback:       int   = 2


class ScannerConfig(BaseModel):
    top_n_coins:           int = 20
    scan_interval_seconds: int = 300
    batch_size:            int = 5
    pre_score_threshold:   int = 55


class ObservabilityConfig(BaseModel):
    signals_log:                str   = 'signals.jsonl'
    closes_log:                 str   = 'closes.jsonl'
    rolling_window_days:        int   = 30
    expectancy_pause_threshold: float = -1.0


class Config(BaseModel):
    signal:        SignalConfig        = SignalConfig()
    risk:          RiskConfig          = RiskConfig()
    smc:           SmcConfig           = SmcConfig()
    scanner:       ScannerConfig       = ScannerConfig()
    observability: ObservabilityConfig = ObservabilityConfig()


def load_config(path: str = 'config.yaml') -> Config:
    try:
        import yaml
        config_path = path if os.path.isabs(path) else os.path.join(
            os.path.dirname(__file__), path
        )
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        return Config(**raw)
    except FileNotFoundError:
        return Config()
    except Exception as e:
        print(f'[Config] Failed to load {path}: {e} — using defaults')
        return Config()


cfg: Config = load_config()
