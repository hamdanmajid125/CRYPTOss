import asyncio
import datetime
import json
import os
import time
from typing import Callable, Dict, List

_STATE_FILE = 'open_trades.json'
_LOG_FILE   = 'trades.log'
_MAX_HOURS  = float(os.getenv('MAX_HOURS_IN_TRADE', '8'))


class TradeManager:
    """
    Stage-machine trade manager.

    Stage 1 — full position open, waiting for TP1 / SL / time-stop / invalidation.
    Stage 2 — 50 % closed at TP1, SL moved to BE+0.1 %, waiting for TP2.
    Stage 3 — 30 % more closed at TP2, SL trailed to TP1, runner waits for TP3.

    Persists state to open_trades.json so restarts resume mid-trade.
    All events logged to trades.log with structured prefixes.
    """

    def __init__(self, weex_client, risk_manager, notifier=None):
        self.weex     = weex_client
        self.risk     = risk_manager
        self.notifier = notifier
        self._trades: List[Dict] = []
        self._load_state()

    # ── LOGGING ────────────────────────────────────────────────────────────────

    def _log(self, msg: str):
        ts   = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
        line = f'[{ts}] [TradeManager] {msg}'
        print(line)
        try:
            with open(_LOG_FILE, 'a') as f:
                f.write(line + '\n')
        except Exception:
            pass

    # ── STATE PERSISTENCE ─────────────────────────────────────────────────────

    def _save_state(self):
        try:
            with open(_STATE_FILE, 'w') as f:
                json.dump(self._trades, f)
        except Exception as e:
            self._log(f'State save failed: {e}')

    def _load_state(self):
        if not os.path.exists(_STATE_FILE):
            return
        try:
            with open(_STATE_FILE) as f:
                self._trades = json.load(f)
            self._log(f'Restored {len(self._trades)} trade(s) from {_STATE_FILE}')
        except Exception as e:
            self._log(f'State load failed: {e}')

    # ── REGISTER ──────────────────────────────────────────────────────────────

    def register(self, trade: Dict):
        qty = trade.get('qty', 0)
        entry = {
            'id':            trade.get('id', ''),
            'symbol':        trade.get('symbol', ''),
            'side':          trade.get('side', 'LONG'),
            'entry':         trade.get('entry', 0),
            'sl':            trade.get('sl', 0),
            'sl_original':   trade.get('sl', 0),
            'tp1':           trade.get('tp1', 0),
            'tp2':           trade.get('tp2', 0),
            'tp3':           trade.get('tp3', 0),
            'qty':           qty,
            'qty_remaining': qty,
            'atr':           float(trade.get('atr', 0) or 0),
            'invalidation':  float(trade.get('invalidation', 0) or 0),
            'stage':         1,
            'opened_at':     trade.get('opened_at', time.time()),
        }
        self._trades.append(entry)
        self._save_state()
        self._log(f'REGISTER | {entry["side"]} {entry["symbol"]} qty={qty}'
                  f' entry={entry["entry"]} id={entry["id"]}')

    # ── MONITOR LOOP ──────────────────────────────────────────────────────────

    async def monitor_loop(self, broadcast_fn: Callable):
        self._log('Monitor started')
        while True:
            await asyncio.sleep(30)
            try:
                await self._check_all(broadcast_fn)
            except Exception as e:
                self._log(f'Monitor error: {e}')

    async def _check_all(self, broadcast_fn: Callable):
        if not self._trades:
            return
        to_remove = []
        for trade in list(self._trades):
            done = await self._check_one(trade, broadcast_fn)
            if done:
                to_remove.append(trade)
        for t in to_remove:
            if t in self._trades:
                self._trades.remove(t)
        if to_remove:
            self._save_state()

    # ── STAGE MACHINE ─────────────────────────────────────────────────────────

    async def _check_one(self, trade: Dict, broadcast_fn: Callable) -> bool:
        tid           = trade['id']
        sym           = trade['symbol']
        side          = trade['side']
        entry         = trade['entry']
        sl            = trade['sl']
        tp1           = trade['tp1']
        tp2           = trade['tp2']
        tp3           = trade['tp3']
        qty_rem       = trade['qty_remaining']
        atr           = trade.get('atr', 0)
        invalidation  = trade.get('invalidation', 0)
        stage         = trade.get('stage', 1)
        is_long       = side == 'LONG'

        try:
            cur = self.weex.get_ticker(sym).get('last', 0)
            if not cur:
                return False
        except Exception:
            return False

        # ── SL hit (all stages) ─────────────────────────────────────────────
        sl_hit = (cur <= sl) if is_long else (cur >= sl)
        if sl_hit:
            pnl = self._pnl(entry, cur, qty_rem, is_long)
            self.risk.record_close(tid, pnl)
            self.weex.close_position(sym, side, qty_rem)
            self._log(f'SL_HIT | {sym} | stage={stage} cur={cur} sl={sl} PnL={pnl:+.2f}')
            await broadcast_fn({'type': 'trade_closed', 'symbol': sym,
                                'reason': 'SL_HIT', 'pnl': round(pnl, 2)})
            if self.notifier:
                self.notifier.trade_closed(sym, side, pnl, 'SL_HIT')
            return True

        # ── Stage 1: time-stop, invalidation, TP1 ──────────────────────────
        if stage == 1:
            age_h = (time.time() - trade['opened_at']) / 3600
            if age_h >= _MAX_HOURS:
                pnl = self._pnl(entry, cur, qty_rem, is_long)
                self.risk.record_close(tid, pnl)
                self.weex.close_position(sym, side, qty_rem)
                self._log(f'TIME_STOP | {sym} | age={age_h:.1f}h PnL={pnl:+.2f}')
                await broadcast_fn({'type': 'trade_closed', 'symbol': sym,
                                    'reason': 'TIME_STOP', 'pnl': round(pnl, 2)})
                if self.notifier:
                    self.notifier.trade_closed(sym, side, pnl, 'TIME_STOP')
                return True

            if invalidation:
                inv_hit = (cur <= invalidation) if is_long else (cur >= invalidation)
                if inv_hit:
                    pnl = self._pnl(entry, cur, qty_rem, is_long)
                    self.risk.record_close(tid, pnl)
                    self.weex.close_position(sym, side, qty_rem)
                    self._log(f'INVALIDATION | {sym} | cur={cur} inv={invalidation} PnL={pnl:+.2f}')
                    await broadcast_fn({'type': 'trade_closed', 'symbol': sym,
                                        'reason': 'INVALIDATION', 'pnl': round(pnl, 2)})
                    if self.notifier:
                        self.notifier.trade_closed(sym, side, pnl, 'INVALIDATION')
                    return True

            tp1_hit = (cur >= tp1) if is_long else (cur <= tp1)
            if tp1 and tp1_hit:
                qty_close = round(qty_rem * 0.50, 6)
                pnl = self._pnl(entry, tp1, qty_close, is_long)
                self.risk.record_partial_pnl(pnl)
                self.weex.close_partial(tid, 0.50, sym, side, exit_price=tp1)
                be_sl = round(entry * 1.001 if is_long else entry * 0.999, 6)
                self.weex.modify_sl(tid, be_sl, sym, side)
                trade['qty_remaining'] = round(qty_rem - qty_close, 6)
                trade['sl']   = be_sl
                trade['stage'] = 2
                self._save_state()
                self._log(f'STAGE_2 | {sym} | TP1={tp1} closed 50% PnL={pnl:+.2f} SL→BE {be_sl}')
                await broadcast_fn({'type': 'tp1_hit', 'symbol': sym,
                                    'pnl': round(pnl, 2), 'new_sl': be_sl})
                if self.notifier:
                    self.notifier.trade_closed(sym, side, pnl, 'TP1_HIT')

        # ── Stage 2: TP2 ────────────────────────────────────────────────────
        elif stage == 2:
            tp2_hit = (cur >= tp2) if is_long else (cur <= tp2)
            if tp2 and tp2_hit:
                qty_close = round(qty_rem * 0.60, 6)
                pnl = self._pnl(entry, tp2, qty_close, is_long)
                self.risk.record_partial_pnl(pnl)
                self.weex.close_partial(tid, 0.60, sym, side, exit_price=tp2)
                self.weex.modify_sl(tid, tp1, sym, side)
                trade['qty_remaining'] = round(qty_rem - qty_close, 6)
                trade['sl']   = tp1
                trade['stage'] = 3
                self._save_state()
                self._log(f'STAGE_3 | {sym} | TP2={tp2} closed 60% of rem PnL={pnl:+.2f} SL→TP1 {tp1}')
                await broadcast_fn({'type': 'tp2_hit', 'symbol': sym,
                                    'pnl': round(pnl, 2), 'new_sl': tp1})
                if self.notifier:
                    self.notifier.trade_closed(sym, side, pnl, 'TP2_HIT')

        # ── Stage 3: runner — TP3 or trail SL ───────────────────────────────
        elif stage == 3:
            tp3_hit = (cur >= tp3) if is_long else (cur <= tp3)
            if tp3 and tp3_hit:
                pnl = self._pnl(entry, tp3, qty_rem, is_long)
                self.risk.record_close(tid, pnl)
                self.weex.close_position(sym, side, qty_rem)
                self._log(f'TP3_HIT | {sym} | PnL={pnl:+.2f}')
                await broadcast_fn({'type': 'tp3_hit', 'symbol': sym, 'pnl': round(pnl, 2)})
                if self.notifier:
                    self.notifier.trade_closed(sym, side, pnl, 'TP3_HIT')
                return True

            if atr > 0:
                cur_sl = trade['sl']
                if is_long:
                    new_sl = round(max(entry + 1.5 * atr, cur - 1.5 * atr), 6)
                    if new_sl > cur_sl:
                        self.weex.modify_sl(tid, new_sl, sym, side)
                        trade['sl'] = new_sl
                        self._save_state()
                        self._log(f'TRAIL_SL | {sym} | {cur_sl} → {new_sl} (cur={cur})')
                else:
                    new_sl = round(min(entry - 1.5 * atr, cur + 1.5 * atr), 6)
                    if new_sl < cur_sl:
                        self.weex.modify_sl(tid, new_sl, sym, side)
                        trade['sl'] = new_sl
                        self._save_state()
                        self._log(f'TRAIL_SL | {sym} | {cur_sl} → {new_sl} (cur={cur})')

        return False

    # ── HELPERS ───────────────────────────────────────────────────────────────

    @staticmethod
    def _pnl(entry: float, exit_price: float, qty: float, is_long: bool) -> float:
        return (exit_price - entry) * qty if is_long else (entry - exit_price) * qty

    def get_open_count(self) -> int:
        return len(self._trades)
