"""Performance metrics for backtests."""
import math
import os
from pathlib import Path
from typing import List

import pandas as pd

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except Exception:
    _HAS_MPL = False


def compute_metrics(trades: List[dict], equity_curve: List[float],
                    initial_capital: float = 10_000.0) -> dict:
    """
    Compute standard performance metrics from a list of closed trades and
    an equity curve (one value per bar, starting at initial_capital).
    """
    n = len(trades)
    if n == 0 or len(equity_curve) < 2:
        return _empty_metrics()

    final_equity = equity_curve[-1]
    total_return = (final_equity - initial_capital) / initial_capital * 100

    # Build daily equity series for Sharpe / Sortino
    eq_s = pd.Series(equity_curve, dtype=float)
    daily = eq_s.resample('1D').last() if hasattr(eq_s.index, 'freq') else eq_s
    daily_ret = daily.pct_change().dropna()

    bars_per_year = 8_760  # 1h bars
    n_bars = len(equity_curve)
    years = max(n_bars / bars_per_year, 1 / 365)
    cagr = ((final_equity / initial_capital) ** (1 / years) - 1) * 100

    ann_factor = math.sqrt(252)  # daily annualisation
    if len(daily_ret) > 1 and daily_ret.std() > 0:
        sharpe  = daily_ret.mean() / daily_ret.std() * ann_factor
        neg_ret = daily_ret[daily_ret < 0]
        sortino = (daily_ret.mean() / neg_ret.std() * ann_factor
                   if len(neg_ret) > 0 and neg_ret.std() > 0 else float('nan'))
    else:
        sharpe = sortino = float('nan')

    # Drawdown
    running_max = eq_s.cummax()
    dd_series   = (eq_s - running_max) / running_max * 100
    max_dd      = float(dd_series.min())
    # DD duration in bars
    in_dd    = dd_series < 0
    dd_len   = 0
    max_dd_dur = 0
    for v in in_dd:
        if v:
            dd_len += 1
            max_dd_dur = max(max_dd_dur, dd_len)
        else:
            dd_len = 0

    # Trade stats
    pnls       = [t.get('pnl', 0.0) for t in trades]
    wins       = [p for p in pnls if p > 0]
    losses     = [p for p in pnls if p <= 0]
    win_rate   = len(wins) / n * 100 if n else 0
    avg_win    = sum(wins)   / len(wins)   if wins   else 0
    avg_loss   = sum(losses) / len(losses) if losses else 0
    expectancy = (win_rate / 100 * avg_win) + ((1 - win_rate / 100) * avg_loss)

    gross_profit = sum(wins)
    gross_loss   = abs(sum(losses))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    return {
        'total_return_pct': round(total_return, 2),
        'cagr_pct':         round(cagr, 2),
        'sharpe':           round(sharpe, 3)  if not math.isnan(sharpe)  else None,
        'sortino':          round(sortino, 3) if not math.isnan(sortino) else None,
        'max_dd_pct':       round(max_dd, 2),
        'max_dd_bars':      max_dd_dur,
        'n_trades':         n,
        'win_rate_pct':     round(win_rate, 1),
        'avg_win_usdt':     round(avg_win,  2),
        'avg_loss_usdt':    round(avg_loss, 2),
        'expectancy_usdt':  round(expectancy, 2),
        'profit_factor':    round(profit_factor, 3),
        'best_trade_usdt':  round(max(pnls), 2) if pnls else 0,
        'worst_trade_usdt': round(min(pnls), 2) if pnls else 0,
        'final_equity':     round(final_equity, 2),
    }


def _empty_metrics() -> dict:
    return {
        'total_return_pct': 0, 'cagr_pct': 0, 'sharpe': None, 'sortino': None,
        'max_dd_pct': 0, 'max_dd_bars': 0, 'n_trades': 0, 'win_rate_pct': 0,
        'avg_win_usdt': 0, 'avg_loss_usdt': 0, 'expectancy_usdt': 0,
        'profit_factor': 0, 'best_trade_usdt': 0, 'worst_trade_usdt': 0,
        'final_equity': 0,
    }


def save_results(
    trades: List[dict],
    equity_curve: List[float],
    metrics: dict,
    out_dir: str,
    label: str,
) -> None:
    """Save trades CSV, equity CSV, and equity PNG to out_dir/label.*"""
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Trades CSV
    if trades:
        pd.DataFrame(trades).to_csv(
            os.path.join(out_dir, f'{label}_trades.csv'), index=False)

    # Equity curve CSV
    eq_df = pd.DataFrame({'equity': equity_curve})
    eq_df.to_csv(os.path.join(out_dir, f'{label}_equity.csv'), index=False)

    # Equity curve PNG
    if _HAS_MPL:
        _, ax = plt.subplots(figsize=(12, 5))
        ax.plot(equity_curve, linewidth=1)
        ax.set_title(f'{label} — equity curve')
        ax.set_xlabel('Bar')
        ax.set_ylabel('Capital (USDT)')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{label}_equity.png'), dpi=120)
        plt.close()


def print_metrics_table(metrics: dict, label: str = '') -> None:
    sep = '─' * 54
    print(f'\n{sep}')
    if label:
        print(f'  {label}')
    print(sep)
    rows = [
        ('Total return',   f"{metrics['total_return_pct']:+.2f} %"),
        ('CAGR',           f"{metrics['cagr_pct']:+.2f} %"),
        ('Sharpe',         str(metrics['sharpe'])),
        ('Sortino',        str(metrics['sortino'])),
        ('Max drawdown',   f"{metrics['max_dd_pct']:.2f} %"),
        ('Max DD bars',    str(metrics['max_dd_bars'])),
        ('Trades',         str(metrics['n_trades'])),
        ('Win rate',       f"{metrics['win_rate_pct']:.1f} %"),
        ('Avg win',        f"${metrics['avg_win_usdt']:+.2f}"),
        ('Avg loss',       f"${metrics['avg_loss_usdt']:.2f}"),
        ('Expectancy',     f"${metrics['expectancy_usdt']:+.2f}"),
        ('Profit factor',  str(metrics['profit_factor'])),
        ('Best trade',     f"${metrics['best_trade_usdt']:+.2f}"),
        ('Worst trade',    f"${metrics['worst_trade_usdt']:.2f}"),
        ('Final equity',   f"${metrics['final_equity']:.2f}"),
    ]
    for name, val in rows:
        print(f'  {name:<20} {val}')
    print(sep)
