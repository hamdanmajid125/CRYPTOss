"""
trade_journal.py — Phase 5
SQLite-backed trade journal. Records every signal, trade open, and close
with full context so you can analyse what setups actually work.
"""
import sqlite3
import time
import json
import os
from typing import Optional


_DB_PATH = os.getenv('JOURNAL_DB', 'trades.db')


class TradeJournal:
    def __init__(self, db_path: str = _DB_PATH):
        self.db_path = db_path
        self._init_db()

    # ── SCHEMA ────────────────────────────────────────────────────────────────

    def _init_db(self):
        with self._conn() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS signals (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts          REAL NOT NULL,
                    symbol      TEXT NOT NULL,
                    action      TEXT NOT NULL,
                    confidence  INTEGER,
                    setup_type  TEXT,
                    entry       REAL,
                    sl          REAL,
                    tp1         REAL,
                    tp2         REAL,
                    tp3         REAL,
                    rr          TEXT,
                    regime      TEXT,
                    bos_type    TEXT,
                    tf_bias     TEXT,
                    fng_value   INTEGER,
                    candle_pat  TEXT,
                    vol_signal  TEXT,
                    pre_score   INTEGER,
                    reason      TEXT,
                    raw_json    TEXT
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS trades (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    signal_id   INTEGER REFERENCES signals(id),
                    order_id    TEXT,
                    ts_open     REAL NOT NULL,
                    ts_close    REAL,
                    symbol      TEXT NOT NULL,
                    side        TEXT NOT NULL,
                    qty         REAL,
                    entry       REAL,
                    sl          REAL,
                    tp1         REAL,
                    exit_price  REAL,
                    pnl_usdt    REAL,
                    exit_reason TEXT,
                    paper       INTEGER DEFAULT 1
                )
            ''')
            conn.execute('''
                CREATE TABLE IF NOT EXISTS daily_stats (
                    date        TEXT PRIMARY KEY,
                    trades      INTEGER DEFAULT 0,
                    wins        INTEGER DEFAULT 0,
                    losses      INTEGER DEFAULT 0,
                    pnl_usdt    REAL DEFAULT 0,
                    best_trade  REAL DEFAULT 0,
                    worst_trade REAL DEFAULT 0
                )
            ''')

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    # ── WRITE ─────────────────────────────────────────────────────────────────

    def log_signal(self, signal: dict) -> int:
        """Insert a signal row and return its rowid."""
        data = signal.get('bos_choch', {})
        with self._conn() as conn:
            cur = conn.execute('''
                INSERT INTO signals
                (ts, symbol, action, confidence, setup_type, entry, sl, tp1, tp2, tp3,
                 rr, regime, bos_type, tf_bias, fng_value, candle_pat, vol_signal,
                 pre_score, reason, raw_json)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ''', (
                time.time(),
                signal.get('symbol', ''),
                signal.get('action', 'WAIT'),
                signal.get('confidence', 0),
                signal.get('setup_type', ''),
                signal.get('entry', 0),
                signal.get('sl', 0),
                signal.get('tp1', 0),
                signal.get('tp2', 0),
                signal.get('tp3', 0),
                signal.get('rr', ''),
                signal.get('regime', ''),
                data.get('type', 'NONE'),
                signal.get('timeframe_bias', ''),
                signal.get('fng_value', 0),
                signal.get('candle_pattern', ''),
                signal.get('volume_signal', ''),
                signal.get('pre_score', 0),
                signal.get('reason', ''),
                json.dumps(signal),
            ))
            return cur.lastrowid

    def log_trade_open(self, signal_id: int, order_id: str, symbol: str,
                       side: str, qty: float, entry: float,
                       sl: float, tp1: float, paper: bool = True) -> int:
        with self._conn() as conn:
            cur = conn.execute('''
                INSERT INTO trades
                (signal_id, order_id, ts_open, symbol, side, qty, entry, sl, tp1, paper)
                VALUES (?,?,?,?,?,?,?,?,?,?)
            ''', (signal_id, order_id, time.time(), symbol, side, qty, entry, sl, tp1, int(paper)))
            return cur.lastrowid

    def log_trade_close(self, trade_id: int, exit_price: float,
                        pnl_usdt: float, exit_reason: str = 'manual'):
        today = time.strftime('%Y-%m-%d', time.gmtime())
        with self._conn() as conn:
            conn.execute('''
                UPDATE trades SET ts_close=?, exit_price=?, pnl_usdt=?, exit_reason=?
                WHERE id=?
            ''', (time.time(), exit_price, pnl_usdt, exit_reason, trade_id))
            conn.execute('''
                INSERT INTO daily_stats (date, trades, wins, losses, pnl_usdt, best_trade, worst_trade)
                VALUES (?,1,?,?,?,?,?)
                ON CONFLICT(date) DO UPDATE SET
                    trades      = trades + 1,
                    wins        = wins   + excluded.wins,
                    losses      = losses + excluded.losses,
                    pnl_usdt    = pnl_usdt + excluded.pnl_usdt,
                    best_trade  = MAX(best_trade,  excluded.best_trade),
                    worst_trade = MIN(worst_trade, excluded.worst_trade)
            ''', (
                today,
                1 if pnl_usdt > 0 else 0,
                1 if pnl_usdt <= 0 else 0,
                pnl_usdt,
                max(pnl_usdt, 0),
                min(pnl_usdt, 0),
            ))

    # ── READ ──────────────────────────────────────────────────────────────────

    def get_recent_signals(self, limit: int = 50) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                'SELECT * FROM signals ORDER BY ts DESC LIMIT ?', (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_open_trades(self) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                'SELECT * FROM trades WHERE ts_close IS NULL ORDER BY ts_open DESC'
            ).fetchall()
        return [dict(r) for r in rows]

    def get_daily_stats(self, days: int = 7) -> list:
        with self._conn() as conn:
            rows = conn.execute(
                'SELECT * FROM daily_stats ORDER BY date DESC LIMIT ?', (days,)
            ).fetchall()
        return [dict(r) for r in rows]

    def get_win_rate(self) -> dict:
        with self._conn() as conn:
            row = conn.execute('''
                SELECT
                    COUNT(*)              AS total,
                    SUM(CASE WHEN pnl_usdt > 0 THEN 1 ELSE 0 END) AS wins,
                    SUM(pnl_usdt)         AS total_pnl,
                    AVG(pnl_usdt)         AS avg_pnl
                FROM trades WHERE ts_close IS NOT NULL
            ''').fetchone()
        total = row['total'] or 0
        wins  = row['wins']  or 0
        return {
            'total_trades': total,
            'wins':         wins,
            'losses':       total - wins,
            'win_rate':     round(wins / total * 100, 1) if total else 0,
            'total_pnl':    round(row['total_pnl'] or 0, 2),
            'avg_pnl':      round(row['avg_pnl']   or 0, 2),
        }

    def get_best_setups(self) -> list:
        """Returns setup types ranked by avg PnL — useful for tuning which setups to take."""
        with self._conn() as conn:
            rows = conn.execute('''
                SELECT s.setup_type,
                       COUNT(t.id)       AS trades,
                       AVG(t.pnl_usdt)   AS avg_pnl,
                       SUM(t.pnl_usdt)   AS total_pnl
                FROM trades t
                JOIN signals s ON t.signal_id = s.id
                WHERE t.ts_close IS NOT NULL AND s.setup_type != ''
                GROUP BY s.setup_type
                ORDER BY avg_pnl DESC
                LIMIT 10
            ''').fetchall()
        return [dict(r) for r in rows]

    def find_trade_by_order_id(self, order_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute(
                'SELECT * FROM trades WHERE order_id=? LIMIT 1', (order_id,)
            ).fetchone()
        return dict(row) if row else None
