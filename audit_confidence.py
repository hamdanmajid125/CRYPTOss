"""
Phase-6: Confidence Calibration Audit

Reads trades.log and backtest_results/*_trades.csv, buckets trades by stated
confidence, computes TP1 hit rate / average R / expectancy per bucket, and
prints sizing recommendations.

Usage:
    python audit_confidence.py [--log trades.log] [--csv backtest_results/]
"""
import argparse
import csv
import os
import re
from collections import defaultdict
from typing import Dict, List


BUCKETS = [(65, 70), (70, 75), (75, 80), (80, 85), (85, 101)]
BUCKET_LABELS = ['65-70', '70-75', '75-80', '80-85', '85+']


# ── PARSE LIVE trades.log ──────────────────────────────────────────────────────

def parse_trades_log(path: str) -> List[dict]:
    """
    Extract completed trade records from trades.log.
    Looks for SIGNAL lines (confidence) paired with STAGE_2/TP1_HIT/SL_HIT lines.
    Returns a list of dicts with: confidence, outcome, pnl_usdt (if available).
    """
    if not os.path.exists(path):
        print(f'[Audit] {path} not found — skipping live log')
        return []

    trades = []
    pending: Dict[str, dict] = {}   # symbol -> last open signal

    with open(path, encoding='utf-8', errors='replace') as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue

            # SIGNAL line: "SIGNAL | LONG | BTC/USDT | conf:82% | ..."
            m = re.search(r'SIGNAL\s*\|\s*(LONG|SHORT)\s*\|\s*([^|]+)\s*\|\s*conf:(\d+)%', line)
            if m:
                sym  = m.group(2).strip()
                conf = int(m.group(3))
                pending[sym] = {'confidence': conf, 'symbol': sym, 'outcome': None, 'pnl': None}
                continue

            # TP1 hit → WIN
            m2 = re.search(r'STAGE_2\s*\|\s*([^|]+)\s*\|.*TP1', line)
            if m2:
                sym = m2.group(1).strip()
                if sym in pending:
                    pending[sym]['outcome'] = 'WIN_TP1'
                    pnl_m = re.search(r'PnL=([+-]?\d+\.?\d*)', line)
                    if pnl_m:
                        pending[sym]['pnl'] = float(pnl_m.group(1))
                    trades.append(dict(pending.pop(sym)))
                continue

            # TP3 hit → WIN big
            m3 = re.search(r'TP3_HIT\s*\|\s*([^|]+)\s*\|', line)
            if m3:
                sym = m3.group(1).strip()
                if sym in pending:
                    pending[sym]['outcome'] = 'WIN_TP3'
                    pnl_m = re.search(r'PnL=([+-]?\d+\.?\d*)', line)
                    if pnl_m:
                        pending[sym]['pnl'] = float(pnl_m.group(1))
                    trades.append(dict(pending.pop(sym)))
                continue

            # SL / TIME_STOP / INVALIDATION → LOSS/NEUTRAL
            for prefix in ('SL_HIT', 'TIME_STOP', 'INVALIDATION'):
                m4 = re.search(rf'{prefix}\s*\|\s*([^|]+)\s*\|', line)
                if m4:
                    sym = m4.group(1).strip()
                    if sym in pending:
                        pending[sym]['outcome'] = prefix
                        pnl_m = re.search(r'PnL=([+-]?\d+\.?\d*)', line)
                        if pnl_m:
                            pending[sym]['pnl'] = float(pnl_m.group(1))
                        trades.append(dict(pending.pop(sym)))
                    break

    print(f'[Audit] Parsed {len(trades)} completed trades from {path}')
    return trades


# ── PARSE BACKTEST CSV ─────────────────────────────────────────────────────────

def parse_backtest_csvs(directory: str) -> List[dict]:
    records = []
    if not os.path.isdir(directory):
        return records
    for fname in os.listdir(directory):
        if not fname.endswith('_trades.csv'):
            continue
        fpath = os.path.join(directory, fname)
        try:
            with open(fpath, newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        conf = int(float(row.get('confidence', 0)))
                        pnl  = float(row.get('pnl_usdt', 0))
                        reason = row.get('exit_reason', '')
                        tp1_hit = reason in ('TP1_HIT', 'TP2_HIT', 'TP3_HIT',
                                             'STAGE_2', 'STAGE_3')
                        records.append({
                            'confidence': conf,
                            'pnl':        pnl,
                            'outcome':    'WIN_TP1' if tp1_hit else reason,
                            'symbol':     row.get('symbol', ''),
                        })
                    except (ValueError, KeyError):
                        continue
        except Exception as e:
            print(f'[Audit] Could not read {fpath}: {e}')
    print(f'[Audit] Loaded {len(records)} trades from backtest CSVs in {directory}')
    return records


# ── BUCKET ANALYSIS ────────────────────────────────────────────────────────────

def bucket_trades(trades: List[dict]) -> dict:
    buckets: Dict[str, List[dict]] = defaultdict(list)
    for t in trades:
        conf = t.get('confidence', 0)
        if conf is None:
            continue
        try:
            conf = int(conf)
        except (TypeError, ValueError):
            continue
        for (lo, hi), label in zip(BUCKETS, BUCKET_LABELS):
            if lo <= conf < hi:
                buckets[label].append(t)
                break
    return dict(buckets)


def analyse_bucket(trades: List[dict]) -> dict:
    if not trades:
        return {'count': 0, 'tp1_hit_rate': 0, 'avg_pnl': 0, 'expectancy': 0}
    tp1_hits   = [t for t in trades if str(t.get('outcome', '')).startswith('WIN')]
    losses     = [t for t in trades if not str(t.get('outcome', '')).startswith('WIN')]
    tp1_rate   = len(tp1_hits) / len(trades) * 100
    pnls       = [t['pnl'] for t in trades if t.get('pnl') is not None]
    avg_pnl    = sum(pnls) / len(pnls) if pnls else 0.0
    win_pnls   = [t['pnl'] for t in tp1_hits if t.get('pnl') is not None]
    loss_pnls  = [abs(t['pnl']) for t in losses if t.get('pnl') is not None]
    avg_win    = sum(win_pnls) / len(win_pnls) if win_pnls else 0.0
    avg_loss   = sum(loss_pnls) / len(loss_pnls) if loss_pnls else 0.0
    wr         = len(tp1_hits) / len(trades)
    expectancy = wr * avg_win - (1 - wr) * avg_loss

    return {
        'count':        len(trades),
        'tp1_hits':     len(tp1_hits),
        'tp1_hit_rate': round(tp1_rate, 1),
        'avg_pnl':      round(avg_pnl, 2),
        'expectancy':   round(expectancy, 2),
        'avg_win':      round(avg_win, 2),
        'avg_loss':     round(avg_loss, 2),
    }


# ── RECOMMENDATIONS ───────────────────────────────────────────────────────────

def recommendations(analysis: dict, current_min_conf: int = 72) -> List[str]:
    recs = []
    best_bucket = None
    best_rate   = 0.0
    worst_bucket = None
    worst_rate   = 100.0

    for label, stats in analysis.items():
        if stats['count'] < 3:
            continue
        rate = stats['tp1_hit_rate']
        if rate > best_rate:
            best_rate, best_bucket = rate, label
        if rate < worst_rate:
            worst_rate, worst_bucket = rate, label

    for label, stats in analysis.items():
        if stats['count'] < 3:
            recs.append(f'  [{label}] Only {stats["count"]} trades — need more data')
            continue
        rate = stats['tp1_hit_rate']
        exp  = stats['expectancy']
        lo   = int(label.split('-')[0].replace('+', ''))

        if rate < 45 and lo < current_min_conf + 10:
            recs.append(
                f'  [{label}] TP1 hit only {rate:.0f}% — '
                f'RAISE MIN_CONFIDENCE to {lo + 5}'
            )
        elif rate >= 70:
            recs.append(
                f'  [{label}] TP1 hit {rate:.0f}% — '
                f'A+ setup, consider 1.5x size when confidence >= {lo}'
            )
        elif exp < 0:
            recs.append(
                f'  [{label}] Negative expectancy (${exp:.2f}) — '
                f'this confidence band loses money, raise threshold'
            )
        else:
            recs.append(
                f'  [{label}] TP1 hit {rate:.0f}% / expectancy ${exp:.2f} — '
                f'acceptable, keep trading'
            )

    if best_bucket:
        recs.append(f'\n  Best bucket: [{best_bucket}] at {best_rate:.0f}% TP1 hit rate')
    if worst_bucket and worst_bucket != best_bucket:
        recs.append(f'  Worst bucket: [{worst_bucket}] at {worst_rate:.0f}% TP1 hit rate')

    return recs


# ── PRINT ──────────────────────────────────────────────────────────────────────

def print_report(all_trades: List[dict]):
    if not all_trades:
        print('[Audit] No trades found — run the bot for a while or run backtest.py first.')
        return

    bucketed = bucket_trades(all_trades)
    analysis = {lbl: analyse_bucket(bucketed.get(lbl, [])) for lbl in BUCKET_LABELS}

    sep = '-' * 72
    print(f'\n{sep}')
    print('  CONFIDENCE CALIBRATION AUDIT')
    print(f'  Total trades analysed: {len(all_trades)}')
    print(sep)
    print(f'  {"Bucket":<10} {"Trades":>7} {"TP1 Hit%":>10} {"Avg PnL":>10} '
          f'{"Expectancy":>12} {"Avg Win":>9} {"Avg Loss":>9}')
    print(sep)
    for lbl in BUCKET_LABELS:
        s = analysis[lbl]
        if s['count'] == 0:
            print(f'  {lbl:<10} {"—":>7}')
            continue
        print(f'  {lbl:<10} {s["count"]:>7} {s["tp1_hit_rate"]:>9.1f}%'
              f' {s["avg_pnl"]:>+10.2f} {s["expectancy"]:>+12.2f}'
              f' {s["avg_win"]:>+9.2f} {s["avg_loss"]:>-9.2f}')
    print(sep)
    print('\n  RECOMMENDATIONS:')
    for rec in recommendations(analysis):
        print(rec)
    print()


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Confidence calibration audit')
    parser.add_argument('--log', default='trades.log',
                        help='Path to trades.log (default: trades.log)')
    parser.add_argument('--csv', default='backtest_results',
                        help='Directory with *_trades.csv files (default: backtest_results)')
    args = parser.parse_args()

    live_trades = parse_trades_log(args.log)
    bt_trades   = parse_backtest_csvs(args.csv)
    all_trades  = live_trades + bt_trades

    print_report(all_trades)


if __name__ == '__main__':
    main()
