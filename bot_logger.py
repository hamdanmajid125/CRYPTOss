"""
Structured logging for the trading bot.

Two layers:
  1. structlog — human-readable + machine-parseable console output.
  2. SignalLogger — appends every signal decision to signals.jsonl so the
     dataset can be used for future model improvements and rolling metrics.
"""
from __future__ import annotations

import json
import os
import datetime
from typing import Optional

import structlog


def setup_logging() -> None:
    """Configure structlog to emit JSON to stdout."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.TimeStamper(fmt='iso', utc=True),
            structlog.stdlib.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.BoundLogger,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = 'bot') -> structlog.BoundLogger:
    return structlog.get_logger(name)


class SignalLogger:
    """
    Writes one JSON line per signal decision to signals.jsonl.
    Each line contains the final action, rule signal, veto decision, and
    a compact features snapshot — the minimum needed for rolling metrics
    and future model training.
    """

    def __init__(self, path: str = 'signals.jsonl'):
        self.path = path
        self._log = get_logger('signal_logger')

    def log_decision(
        self,
        symbol: str,
        market_data: dict,
        rule_sig: dict,
        veto: Optional[dict] = None,
        final_sig: Optional[dict] = None,
    ) -> None:
        if final_sig is None:
            final_sig = rule_sig

        entry = {
            'ts':               datetime.datetime.utcnow().isoformat() + 'Z',
            'symbol':           symbol,
            'action':           final_sig.get('action', 'WAIT'),
            'confidence':       final_sig.get('confidence', 0),
            'rule_action':      rule_sig.get('action', 'WAIT'),
            'rule_confidence':  rule_sig.get('confidence', 0),
            'veto_decision':    veto.get('decision') if veto else None,
            'veto_adj':         veto.get('confidence_adjustment', 0) if veto else 0,
            'reason':           final_sig.get('reason') or rule_sig.get('reason', ''),
            'features': {
                'price':       market_data.get('price'),
                'atr':         market_data.get('atr'),
                'rsi':         market_data.get('rsi'),
                'adx':         market_data.get('adx'),
                'bb_pct':      market_data.get('bb_width_percentile'),
                'mtf':         market_data.get('timeframe_bias', {}).get('overall'),
                'agreement':   market_data.get('timeframe_bias', {}).get('agreement'),
                'volume':      market_data.get('volume_signal'),
                'fng':         market_data.get('fear_greed', {}).get('value'),
                'fr_rate':     market_data.get('funding_rate', {}).get('rate'),
                'oi_compressed': market_data.get('open_interest', {}).get('compressed'),
            },
        }

        try:
            with open(self.path, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry, default=str) + '\n')
        except Exception as e:
            self._log.error('signal_log_write_failed', error=str(e))

    def read_recent(self, days: int = 30) -> list[dict]:
        """Return all entries from the last `days` calendar days."""
        cutoff = (datetime.datetime.utcnow()
                  - datetime.timedelta(days=days)).isoformat() + 'Z'
        entries: list[dict] = []
        if not os.path.exists(self.path):
            return entries
        try:
            with open(self.path, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if rec.get('ts', '') >= cutoff:
                            entries.append(rec)
                    except json.JSONDecodeError:
                        pass
        except Exception:
            pass
        return entries
