import time
import asyncio
from typing import Dict, List, Callable, Optional


class TradeManager:
    """
    Monitors all open trades every 30s.
    Checks TP1/TP2/TP3 partial closes, SL hits, and 48-hour hard timeout.
    Works alongside paper positions in WeexClient and live positions on exchange.
    """

    def __init__(self, weex_client, risk_manager, notifier=None):
        self.weex     = weex_client
        self.risk     = risk_manager
        self.notifier = notifier
        self._trades: List[Dict] = []

    def register(self, trade: Dict):
        """Register a newly opened trade for monitoring."""
        self._trades.append({
            **trade,
            'tp1_hit':   False,
            'tp2_hit':   False,
            'opened_at': trade.get('opened_at', time.time()),
        })
        print(f'[TradeManager] Registered {trade.get("side")} {trade.get("symbol")} id={trade.get("id")}')

    async def monitor_loop(self, broadcast_fn: Callable):
        """Background coroutine — runs every 30s checking all open trades."""
        print('[TradeManager] Monitor started')
        while True:
            await asyncio.sleep(30)
            try:
                await self._check_all(broadcast_fn)
            except Exception as e:
                print(f'[TradeManager] Error: {e}')

    async def _check_all(self, broadcast_fn: Callable):
        if not self._trades:
            return

        to_remove = []
        for trade in self._trades:
            done = await self._check_one(trade, broadcast_fn)
            if done:
                to_remove.append(trade)

        for t in to_remove:
            if t in self._trades:
                self._trades.remove(t)

    async def _check_one(self, trade: Dict, broadcast_fn: Callable) -> bool:
        """Returns True if the trade is fully closed and should be removed."""
        sym   = trade.get('symbol', '')
        side  = trade.get('side', 'LONG')
        entry = trade.get('entry', 0)
        sl    = trade.get('sl', 0)
        tp1   = trade.get('tp1', 0)
        tp2   = trade.get('tp2', 0)
        tp3   = trade.get('tp3', 0)
        qty   = trade.get('qty', 0)
        tid   = trade.get('id', '')

        try:
            ticker = self.weex.get_ticker(sym)
            cur    = ticker.get('last', 0)
            if not cur:
                return False
        except Exception:
            return False

        is_long = side == 'LONG'
        age_h   = (time.time() - trade['opened_at']) / 3600

        # 48-hour hard timeout — close at market
        if age_h >= 48:
            pnl = self._calc_pnl(entry, cur, qty, is_long)
            self._record_close(tid, pnl)
            await broadcast_fn({'type': 'trade_closed', 'symbol': sym,
                                'reason': 'TIMEOUT_48H', 'pnl': round(pnl, 2)})
            if self.notifier:
                self.notifier.trade_closed(sym, side, pnl, 'TIMEOUT_48H')
            print(f'[TradeManager] TIMEOUT_48H {sym} PnL={pnl:+.2f}')
            return True

        # SL hit
        sl_hit = (cur <= sl) if is_long else (cur >= sl)
        if sl_hit:
            pnl = self._calc_pnl(entry, cur, qty, is_long)
            self._record_close(tid, pnl)
            await broadcast_fn({'type': 'trade_closed', 'symbol': sym,
                                'reason': 'SL_HIT', 'pnl': round(pnl, 2)})
            if self.notifier:
                self.notifier.trade_closed(sym, side, pnl, 'SL_HIT')
            print(f'[TradeManager] SL_HIT {sym} PnL={pnl:+.2f}')
            return True

        # TP1 hit (40% of position)
        tp1_hit = (cur >= tp1) if is_long else (cur <= tp1)
        if tp1 and tp1_hit and not trade['tp1_hit']:
            trade['tp1_hit'] = True
            partial_pnl = self._calc_pnl(entry, tp1, qty * 0.40, is_long)
            trade['sl'] = entry  # move SL to breakeven
            await broadcast_fn({'type': 'tp1_hit', 'symbol': sym,
                                'pnl': round(partial_pnl, 2), 'new_sl': entry})
            if self.notifier:
                self.notifier.trade_closed(sym, side, partial_pnl, 'TP1_HIT')
            print(f'[TradeManager] TP1_HIT {sym} partial PnL={partial_pnl:+.2f}, SL→BE')

        # TP2 hit (40% of position, only after TP1)
        tp2_hit = (cur >= tp2) if is_long else (cur <= tp2)
        if tp2 and tp2_hit and trade['tp1_hit'] and not trade['tp2_hit']:
            trade['tp2_hit'] = True
            partial_pnl = self._calc_pnl(entry, tp2, qty * 0.40, is_long)
            trade['sl'] = tp1  # trail SL to TP1
            await broadcast_fn({'type': 'tp2_hit', 'symbol': sym,
                                'pnl': round(partial_pnl, 2), 'new_sl': tp1})
            if self.notifier:
                self.notifier.trade_closed(sym, side, partial_pnl, 'TP2_HIT')
            print(f'[TradeManager] TP2_HIT {sym} partial PnL={partial_pnl:+.2f}, SL→TP1')

        # TP3 hit — runner (20% of position, only after TP2)
        tp3_hit = (cur >= tp3) if is_long else (cur <= tp3)
        if tp3 and tp3_hit and trade['tp2_hit']:
            runner_pnl = self._calc_pnl(entry, tp3, qty * 0.20, is_long)
            self._record_close(tid, runner_pnl)
            await broadcast_fn({'type': 'tp3_hit', 'symbol': sym,
                                'pnl': round(runner_pnl, 2)})
            if self.notifier:
                self.notifier.trade_closed(sym, side, runner_pnl, 'TP3_HIT')
            print(f'[TradeManager] TP3_HIT {sym} runner PnL={runner_pnl:+.2f} — trade complete')
            return True

        return False

    @staticmethod
    def _calc_pnl(entry: float, exit_price: float, qty: float, is_long: bool) -> float:
        if is_long:
            return (exit_price - entry) * qty
        return (entry - exit_price) * qty

    def _record_close(self, trade_id: str, pnl: float):
        self.risk.record_close(trade_id, pnl)

    def get_open_count(self) -> int:
        return len(self._trades)
