"""backtest package — event-driven backtester for the crypto trading bot."""

from backtest.engine import BarReplayer  # noqa: F401

# Backwards-compat alias: main.py imports Backtester from backtest
Backtester = BarReplayer
