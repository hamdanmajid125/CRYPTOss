import ccxt
import time
import os
from typing import Optional, List, Dict


class WeexClient:
    def __init__(self, api_key: str, secret: str, passphrase: str, paper: bool = True):
        self.paper = paper
        self.exchange = ccxt.weex({
            'apiKey': api_key,
            'secret': secret,
            'password': passphrase,
            'options': {'defaultType': 'swap'},
        })
        # Public Binance for price feeds (works in both paper and live mode)
        self._public = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})

        # Paper trading state
        self._paper_balance = float(os.getenv('ACCOUNT_USDT', '10000'))
        self._paper_positions: List[Dict] = []
        self._paper_order_id = 1000

        print(f'[WEEX] Initialized — paper={paper}')

    # ── BALANCE ────────────────────────────────────────────────────────────────

    def get_balance(self) -> Dict:
        if self.paper:
            used = sum(p.get('margin', 0) for p in self._paper_positions)
            unreal_pnl = sum(p.get('unrealized_pnl', 0) for p in self._paper_positions)
            return {
                'USDT': {
                    'free': round(self._paper_balance - used, 2),
                    'used': round(used, 2),
                    'total': round(self._paper_balance, 2),
                },
                'unrealized_pnl': round(unreal_pnl, 2),
                'paper': True,
            }
        try:
            raw = self.exchange.fetch_balance()
            usdt = raw.get('USDT', {})
            positions = self.get_positions()
            unreal_pnl = sum(float(p.get('unrealizedPnl', 0)) for p in positions)
            return {
                'USDT': {
                    'free': usdt.get('free', 0),
                    'used': usdt.get('used', 0),
                    'total': usdt.get('total', 0),
                },
                'unrealized_pnl': round(unreal_pnl, 2),
                'paper': False,
            }
        except Exception as e:
            print(f'[WEEX] Balance fetch failed: {e}')
            return {'error': str(e), 'USDT': {'free': 0, 'used': 0, 'total': 0}}

    # ── POSITIONS ──────────────────────────────────────────────────────────────

    def get_positions(self) -> List[Dict]:
        if self.paper:
            self._update_paper_pnl()
            return list(self._paper_positions)
        try:
            positions = self.exchange.fetch_positions()
            return [p for p in positions if float(p.get('contracts', 0)) != 0]
        except Exception as e:
            print(f'[WEEX] Positions fetch failed: {e}')
            return []

    def _update_paper_pnl(self):
        for pos in self._paper_positions:
            try:
                ticker = self._public.fetch_ticker(pos['symbol'])
                current = ticker['last']
                entry = pos['entry']
                qty = pos['qty']
                if pos['side'] == 'LONG':
                    pos['unrealized_pnl'] = round((current - entry) * qty, 2)
                    pos['pnl_pct'] = round((current - entry) / entry * 100, 3)
                else:
                    pos['unrealized_pnl'] = round((entry - current) * qty, 2)
                    pos['pnl_pct'] = round((entry - current) / entry * 100, 3)
                pos['current_price'] = current
            except Exception:
                pass

    # ── TICKER ─────────────────────────────────────────────────────────────────

    def get_ticker(self, symbol: str) -> dict:
        try:
            return self._public.fetch_ticker(symbol)
        except Exception as e:
            print(f'[Ticker] Error {symbol}: {e}')
            return {'last': 0, 'bid': 0, 'ask': 0}

    # ── ORDER PLACEMENT ────────────────────────────────────────────────────────

    def place_order(self, symbol: str, side: str, qty: float,
                    entry: float, sl: float, tp1: float,
                    tp2: float = 0, tp3: float = 0) -> Optional[Dict]:
        if self.paper:
            return self._paper_place_order(symbol, side, qty, entry, sl, tp1, tp2, tp3)
        return self._live_place_order(symbol, side, qty, entry, sl, tp1, tp2)

    def _paper_place_order(self, symbol: str, side: str, qty: float,
                           entry: float, sl: float, tp1: float,
                           tp2: float = 0, tp3: float = 0) -> Dict:
        order_id = f'paper_{self._paper_order_id}'
        self._paper_order_id += 1

        # 50% at TP1, 30% at TP2, 20% runner
        qty_tp1 = round(qty * 0.50, 6)
        qty_tp2 = round(qty * 0.30, 6)
        qty_runner = round(qty * 0.20, 6)
        margin = round(entry * qty * 0.10, 2)  # approximate 10x leverage margin

        position = {
            'id': order_id,
            'symbol': symbol,
            'side': side,
            'entry': entry,
            'qty': qty,
            'qty_tp1': qty_tp1,
            'qty_tp2': qty_tp2,
            'qty_runner': qty_runner,
            'sl': sl,
            'tp1': tp1,
            'tp2': tp2 or round(entry + (tp1 - entry) * 1.8, 2),
            'tp3': tp3 or round(entry + (tp1 - entry) * 3.0, 2),
            'current_price': entry,
            'unrealized_pnl': 0.0,
            'pnl_pct': 0.0,
            'margin': margin,
            'sl_moved_to_be': False,
            'sl_moved_to_tp1': False,
            'tp1_hit': False,
            'tp2_hit': False,
            'time': time.time(),
            'status': 'open',
        }
        self._paper_positions.append(position)

        print(f'[PAPER] {side} {qty} {symbol} @ {entry} | SL:{sl} TP1:{tp1} TP2:{position["tp2"]} TP3:{position["tp3"]}')
        return {
            'id': order_id,
            'status': 'paper_filled',
            'symbol': symbol,
            'side': side,
            'qty': qty,
            'entry': entry,
            'sl': sl,
            'tp1': tp1,
        }

    def _live_place_order(self, symbol: str, side: str, qty: float,
                          entry: float, sl: float, tp1: float,
                          tp2: float = 0) -> Optional[Dict]:
        try:
            # Limit entry near current price (0.05% offset for fast fill)
            ccxt_side = 'buy' if side == 'LONG' else 'sell'
            limit_price = round(entry * 0.9995 if side == 'LONG' else entry * 1.0005, 2)

            order = self.exchange.create_order(
                symbol=symbol,
                type='limit',
                side=ccxt_side,
                amount=qty,
                price=limit_price,
            )

            # Place SL as stop-market immediately
            try:
                self.exchange.create_order(
                    symbol=symbol,
                    type='stop_market',
                    side='sell' if side == 'LONG' else 'buy',
                    amount=qty,
                    price=sl,
                    params={'stopPrice': sl, 'reduceOnly': True},
                )
            except Exception as e:
                print(f'[WEEX] SL placement failed: {e}')

            # TP1 limit for 50% of position
            try:
                qty_tp1 = round(qty * 0.50, 6)
                self.exchange.create_order(
                    symbol=symbol,
                    type='limit',
                    side='sell' if side == 'LONG' else 'buy',
                    amount=qty_tp1,
                    price=tp1,
                    params={'reduceOnly': True},
                )
            except Exception as e:
                print(f'[WEEX] TP1 placement failed: {e}')

            # TP2 limit for 30% of position
            if tp2:
                try:
                    qty_tp2 = round(qty * 0.30, 6)
                    self.exchange.create_order(
                        symbol=symbol,
                        type='limit',
                        side='sell' if side == 'LONG' else 'buy',
                        amount=qty_tp2,
                        price=tp2,
                        params={'reduceOnly': True},
                    )
                except Exception as e:
                    print(f'[WEEX] TP2 placement failed: {e}')

            return order
        except Exception as e:
            print(f'[WEEX] Order failed: {e}')
            return None

    # ── POSITION MONITORING ────────────────────────────────────────────────────

    def check_positions(self) -> List[Dict]:
        """Check all paper positions for TP/SL hits and move stops. Returns events."""
        if not self.paper:
            return []
        events = []
        self._update_paper_pnl()
        closed = []
        for pos in self._paper_positions:
            sym = pos['symbol']
            cur = pos.get('current_price', pos['entry'])
            side = pos['side']
            entry = pos['entry']

            is_long = side == 'LONG'
            above = lambda level: cur >= level
            below = lambda level: cur <= level

            hit_sl = below(pos['sl']) if is_long else above(pos['sl'])
            if hit_sl:
                pnl = pos['unrealized_pnl']
                self._paper_balance += pnl
                events.append({'event': 'SL_HIT', 'symbol': sym, 'pnl': pnl, 'pos': pos})
                closed.append(pos)
                print(f'[PAPER] SL hit {sym} — PnL: {pnl:.2f} USDT')
                continue

            hit_tp1 = above(pos['tp1']) if is_long else below(pos['tp1'])
            if hit_tp1 and not pos['tp1_hit']:
                pos['tp1_hit'] = True
                tp1_pnl = round((pos['tp1'] - entry if is_long else entry - pos['tp1']) * pos['qty_tp1'], 2)
                self._paper_balance += tp1_pnl
                if not pos['sl_moved_to_be']:
                    pos['sl'] = entry  # move SL to breakeven
                    pos['sl_moved_to_be'] = True
                events.append({'event': 'TP1_HIT', 'symbol': sym, 'pnl': tp1_pnl, 'new_sl': pos['sl']})
                print(f'[PAPER] TP1 hit {sym} — PnL: +{tp1_pnl:.2f}, SL moved to BE')

            hit_tp2 = above(pos['tp2']) if is_long else below(pos['tp2'])
            if hit_tp2 and not pos['tp2_hit'] and pos['tp1_hit']:
                pos['tp2_hit'] = True
                tp2_pnl = round((pos['tp2'] - entry if is_long else entry - pos['tp2']) * pos['qty_tp2'], 2)
                self._paper_balance += tp2_pnl
                pos['sl'] = pos['tp1']  # trail SL to TP1
                pos['sl_moved_to_tp1'] = True
                events.append({'event': 'TP2_HIT', 'symbol': sym, 'pnl': tp2_pnl, 'new_sl': pos['sl']})
                print(f'[PAPER] TP2 hit {sym} — PnL: +{tp2_pnl:.2f}, SL trailed to TP1')

            atr_warn = pos.get('atr_warn_threshold', 0)
            if atr_warn and not hit_sl:
                adverse = below(entry - atr_warn) if is_long else above(entry + atr_warn)
                if adverse:
                    events.append({'event': 'ADVERSE_MOVE_WARNING', 'symbol': sym, 'current': cur})

        for pos in closed:
            self._paper_positions.remove(pos)
        return events

    def close_position(self, symbol: str, side: str, qty: float) -> Dict:
        if self.paper:
            to_close = [p for p in self._paper_positions if p['symbol'] == symbol and p['side'] == side]
            for pos in to_close:
                self._update_paper_pnl()
                self._paper_balance += pos.get('unrealized_pnl', 0)
                self._paper_positions.remove(pos)
            print(f'[PAPER] Closed {side} {qty} {symbol}')
            return {'status': 'paper_closed', 'symbol': symbol}
        try:
            close_side = 'sell' if side == 'LONG' else 'buy'
            return self.exchange.create_order(symbol, 'market', close_side, qty,
                                              params={'reduceOnly': True})
        except Exception as e:
            print(f'[WEEX] Close failed: {e}')
            return {'error': str(e)}

    def close_all(self) -> List[Dict]:
        results = []
        if self.paper:
            self._update_paper_pnl()
            for pos in list(self._paper_positions):
                pnl = pos.get('unrealized_pnl', 0)
                self._paper_balance += pnl
                results.append({'symbol': pos['symbol'], 'side': pos['side'], 'pnl': pnl})
            self._paper_positions.clear()
            print(f'[PAPER] All positions closed. Balance: {self._paper_balance:.2f}')
            return results
        try:
            positions = self.get_positions()
            for p in positions:
                sym = p.get('symbol')
                side = 'LONG' if float(p.get('contracts', 0)) > 0 else 'SHORT'
                qty = abs(float(p.get('contracts', 0)))
                if qty > 0:
                    res = self.close_position(sym, side, qty)
                    results.append({'symbol': sym, 'result': res})
        except Exception as e:
            print(f'[WEEX] close_all error: {e}')
        return results
