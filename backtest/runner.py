"""
CLI entry point for the backtester.

Usage:
    python -m backtest.runner --symbol BTC/USDT --start 2024-01-01 --end 2025-12-01 --mode rules --capital 10000
    python -m backtest.runner --symbol BTC/USDT --start 2024-01-01 --mode hybrid --walk-forward
"""
import argparse
import os
import sys

# Allow running from the project root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from backtest.data_loader import load_ohlcv
from backtest.engine import BarReplayer
from backtest.metrics import compute_metrics, print_metrics_table, save_results

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def _label(symbol: str, mode: str, start: str, end: str) -> str:
    return f"{symbol.replace('/', '_')}_{mode}_{start}_{end}"


def run_single(
    symbol: str, start: str, end: str, mode: str,
    capital: float, risk_pct: float, agent=None,
) -> dict:
    df = load_ohlcv(symbol, timeframe='1h', start=start, end=end)
    replayer = BarReplayer(
        symbol=symbol, initial_capital=capital,
        risk_pct=risk_pct, mode=mode, agent=agent,
    )
    trades, equity = replayer.run(df)
    metrics = compute_metrics(trades, equity, initial_capital=capital)
    label   = _label(symbol, mode, start, end)

    print_metrics_table(metrics, label=f'{symbol} | {mode} | {start} to {end}')
    save_results(trades, equity, metrics, out_dir=RESULTS_DIR, label=label)
    return metrics


def run_walk_forward(
    symbol: str, start: str, end: str, mode: str,
    capital: float, risk_pct: float, agent=None,
) -> dict:
    df = load_ohlcv(symbol, timeframe='1h', start=start, end=end)
    replayer = BarReplayer(
        symbol=symbol, initial_capital=capital,
        risk_pct=risk_pct, mode=mode, agent=agent,
    )
    trades, equity = replayer.walk_forward(df)
    metrics = compute_metrics(trades, equity, initial_capital=capital)
    label   = _label(symbol, mode + '_wf', start, end)

    print_metrics_table(metrics, label=f'{symbol} | {mode} WALK-FORWARD | {start} to {end}')
    save_results(trades, equity, metrics, out_dir=RESULTS_DIR, label=label)
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='Crypto trading bot backtester',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--symbol',       default='BTC/USDT')
    parser.add_argument('--start',        default='2024-01-01')
    parser.add_argument('--end',          default=None,
                        help='End date (YYYY-MM-DD). Defaults to today.')
    parser.add_argument('--mode',         default='rules',
                        choices=['rules', 'llm', 'hybrid'])
    parser.add_argument('--capital',      type=float, default=10_000.0)
    parser.add_argument('--risk-pct',     type=float, default=1.0,
                        dest='risk_pct')
    parser.add_argument('--walk-forward', action='store_true')
    parser.add_argument('--all-symbols',  action='store_true',
                        help='Run all 4 symbols (BTC, ETH, SOL, BNB) in all modes')
    args = parser.parse_args()

    import pandas as pd
    if args.end is None:
        args.end = pd.Timestamp.utcnow().strftime('%Y-%m-%d')

    agent = None
    if args.mode in ('llm', 'hybrid'):
        try:
            import os
            from claude_agent import ClaudeAgent
            agent = ClaudeAgent(api_key=os.environ['ANTHROPIC_API_KEY'])
        except Exception as e:
            print(f'[Runner] Cannot load ClaudeAgent ({e}). Falling back to rules mode.')
            args.mode = 'rules'

    if args.all_symbols:
        symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT']
        modes   = ['rules', 'llm', 'hybrid'] if agent else ['rules']
        summary = []
        for sym in symbols:
            for m in modes:
                try:
                    ag = agent if m in ('llm', 'hybrid') else None
                    fn = run_walk_forward if args.walk_forward else run_single
                    metrics = fn(sym, args.start, args.end, m, args.capital,
                                 args.risk_pct, agent=ag)
                    summary.append({'symbol': sym, 'mode': m, **metrics})
                except Exception as e:
                    print(f'[Runner] {sym} {m} failed: {e}')

        # Print summary table
        print('\n' + '=' * 80)
        print('  SUMMARY - ALL SYMBOLS x MODES')
        print('=' * 80)
        hdr = f"  {'Symbol':<12} {'Mode':<8} {'Return%':>9} {'Sharpe':>8} {'MaxDD%':>8} {'Trades':>7} {'WinRate%':>9}"
        print(hdr)
        print('-' * 80)
        for r in summary:
            print(f"  {r['symbol']:<12} {r['mode']:<8} "
                  f"{r['total_return_pct']:>+8.2f}% "
                  f"{str(r['sharpe']):>8} "
                  f"{r['max_dd_pct']:>7.2f}% "
                  f"{r['n_trades']:>7} "
                  f"{r['win_rate_pct']:>8.1f}%")
        print('=' * 80)
        return

    fn = run_walk_forward if args.walk_forward else run_single
    fn(args.symbol, args.start, args.end, args.mode, args.capital,
       args.risk_pct, agent=agent)


if __name__ == '__main__':
    main()
