import base64
import ccxt
import hashlib
import hmac
import os
import requests
import time
import uuid
import json
from typing import Optional, List, Dict
from utils import with_retry


class WeexClient:
    BASE_URL = 'https://api-contract.weex.com'

    def __init__(self, api_key: str, secret: str, passphrase: str,
                 paper: bool = True, paper_balance: float = 10000.0):
        self.paper      = paper
        self.api_key    = api_key.strip()
        self.secret     = secret.strip()
        self.passphrase = passphrase.strip()

        self.exchange = ccxt.weex({
            'apiKey':   api_key,
            'secret':   secret,
            'password': passphrase,
            'options':  {'defaultType': 'swap'},
        })
        # Public Binance for price feeds (works in both paper and live mode)
        self._public = ccxt.binance({'enableRateLimit': True, 'options': {'defaultType': 'spot'}})

        # Paper trading state — seeded from caller (usually ACCOUNT_USDT env var)
        self._paper_balance    = paper_balance
        self._paper_positions: List[Dict] = []

        print(f'[WEEX] Initialized — paper={paper}, paper_balance={paper_balance:.2f}')

    # ── RAW HTTP LAYER (HMAC-SHA256) ───────────────────────────────────────────

    def _sign(self, timestamp: str, method: str, path: str, body: str = '') -> str:
        """HMAC-SHA256 base64 of: timestamp + METHOD + path + body."""
        message = timestamp + method.upper() + path + body
        raw = hmac.new(
            self.secret.encode('utf-8'),
            message.encode('utf-8'),
            hashlib.sha256,
        ).digest()
        return base64.b64encode(raw).decode('utf-8')

    def _request(self, method: str, path: str, data: Optional[Dict] = None) -> Optional[Dict]:
        """Direct signed HTTP call. Returns parsed JSON or None."""
        ts   = str(int(time.time() * 1000))
        body = json.dumps(data) if data else ''
        sign = self._sign(ts, method, path, body)
        headers = {
            'ACCESS-KEY':        self.api_key,
            'ACCESS-SIGN':       sign,
            'ACCESS-TIMESTAMP':  ts,
            'ACCESS-PASSPHRASE': self.passphrase,
            'Content-Type':      'application/json',
        }
        url = self.BASE_URL + path
        try:
            if method.upper() == 'GET':
                resp = requests.get(url, headers=headers, timeout=10)
            else:
                resp = requests.post(url, headers=headers, data=body, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            print(f'[Weex] HTTP {method} {path} failed: {type(e).__name__}: {e}')
            import traceback; traceback.print_exc()
            return None

    def _get(self, path: str) -> Optional[Dict]:
        return self._request('GET', path)

    def _post(self, path: str, data: Optional[Dict] = None) -> Optional[Dict]:
        return self._request('POST', path, data)

    # ── CONNECTION TEST ────────────────────────────────────────────────────────

    def test_connection(self) -> dict:
        """Test API credentials and connectivity. Returns status dict."""
        if self.paper:
            return {'status': 'paper_mode', 'connected': True,
                    'message': 'Paper trading — no real API needed'}
        try:
            result = self._get('/capi/v3/account/balance')
            if result is None:
                return {'status': 'error', 'connected': False, 'response': None}
            # API may return a list of balances or a dict with a 'balances' key
            if isinstance(result, list):
                return {'status': 'ok', 'connected': True, 'balance': result}
            if isinstance(result, dict) and 'msg' not in result:
                return {'status': 'ok', 'connected': True, 'balance': result.get('balances', result)}
            return {'status': 'error', 'connected': False, 'response': result}
        except Exception as e:
            return {'status': 'exception', 'connected': False, 'error': str(e)}

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
            print(f'[Weex] FULL ERROR: {type(e).__name__}: {e}')
            import traceback; traceback.print_exc()
            return {'error': str(e), 'USDT': {'free': 0, 'used': 0, 'total': 0}}

    def get_usdt_available(self) -> float:
        """Return available USDT as a single float for risk sizing (paper and live)."""
        if self.paper:
            # Try the real exchange balance so paper sizing reflects the actual wallet.
            try:
                raw  = self.exchange.fetch_balance()
                usdt = raw.get('USDT', {})
                free = float(usdt.get('free', 0) or 0)
                if free > 0:
                    self._paper_balance = free  # keep paper balance in sync with real wallet
                    return free
            except Exception:
                pass
            # Fall back to simulated paper balance minus open-position margin
            used = sum(p.get('margin', 0) for p in self._paper_positions)
            return max(0.0, round(self._paper_balance - used, 2))
        # Live mode
        try:
            bal  = self.get_balance()
            usdt = bal.get('USDT', {})
            free = float(usdt.get('free', 0) or 0)
            if free > 0:
                return free
            return float(usdt.get('total', 0) or 0)
        except Exception as e:
            print(f'[Weex] get_usdt_available failed: {e}')
            return 0.0

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
        order_id = f'paper_{uuid.uuid4().hex[:8]}'

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

        print(f'[PAPER] {side} {qty} {symbol} @ {entry} | SL:{sl} TP1:{tp1} TP2:{position["tp2"]} TP3:{position["tp3"]} id:{order_id}')

        # JSON record to paper_trades.log
        try:
            record = {
                'id': order_id, 'symbol': symbol, 'side': side,
                'qty': qty, 'entry': entry, 'sl': sl,
                'tp1': tp1, 'tp2': position['tp2'], 'tp3': position['tp3'],
                'time': time.strftime('%Y-%m-%d %H:%M:%S'),
            }
            with open('paper_trades.log', 'a') as f:
                f.write(json.dumps(record) + '\n')
        except Exception:
            pass

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

    # Maximum allowed slippage from intended entry (0.15%)
    _MAX_SLIPPAGE_PCT = 0.0015

    def _check_slippage(self, symbol: str, intended_entry: float, side: str) -> tuple:
        """
        Fetches live bid/ask and checks whether the fill would exceed the
        max slippage threshold. Returns (ok: bool, live_price: float).
        """
        try:
            ticker = self._public.fetch_ticker(symbol)
            live   = ticker['ask'] if side == 'LONG' else ticker['bid']
            slip   = abs(live - intended_entry) / intended_entry
            if slip > self._MAX_SLIPPAGE_PCT:
                print(f'[WEEX] Slippage {slip*100:.3f}% > {self._MAX_SLIPPAGE_PCT*100:.2f}% — order rejected')
                return False, live
            return True, live
        except Exception:
            return True, intended_entry   # if check fails, allow order (don't block)

    def _live_place_order(self, symbol: str, side: str, qty: float,
                          entry: float, sl: float, tp1: float,
                          tp2: float = 0) -> Optional[Dict]:
        try:
            # Slippage gate — reject if market moved too far from signal entry
            ok, live_price = self._check_slippage(symbol, entry, side)
            if not ok:
                return None

            # Limit entry near current price (0.05% offset for fast fill)
            ccxt_side = 'buy' if side == 'LONG' else 'sell'
            limit_price = round(live_price * 0.9995 if side == 'LONG' else live_price * 1.0005, 2)

            order = with_retry(lambda: self.exchange.create_order(
                symbol=symbol,
                type='limit',
                side=ccxt_side,
                amount=qty,
                price=limit_price,
            ))

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
            print(f'[Weex] FULL ERROR: {type(e).__name__}: {e}')
            import traceback; traceback.print_exc()
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

    def close_partial(self, order_id: str, fraction: float,
                      symbol: str = '', side: str = '',
                      exit_price: float = 0.0) -> Dict:
        """Close fraction of a position at market. Paper uses exit_price (or current_price) for PnL."""
        if self.paper:
            pos = next((p for p in self._paper_positions if p['id'] == order_id), None)
            if not pos:
                pos = next((p for p in self._paper_positions
                            if p['symbol'] == symbol and p['side'] == side), None)
            if not pos:
                return {'error': 'position_not_found'}
            fill = exit_price or pos.get('current_price', pos['entry'])
            qty_close = round(pos['qty'] * fraction, 6)
            is_long = pos['side'] == 'LONG'
            pnl = round(((fill - pos['entry']) * qty_close if is_long
                         else (pos['entry'] - fill) * qty_close), 2)
            self._paper_balance += pnl
            pos['qty'] = round(pos['qty'] - qty_close, 6)
            pos['margin'] = round(pos['margin'] * (1 - fraction), 2)
            print(f'[PAPER] Partial close {fraction*100:.0f}% {pos["side"]} {pos["symbol"]}'
                  f' @ {fill} PnL={pnl:+.2f}')
            return {'status': 'paper_partial', 'pnl': pnl, 'qty_closed': qty_close}
        try:
            close_side = 'sell' if side == 'LONG' else 'buy'
            positions = self.exchange.fetch_positions([symbol]) if symbol else []
            pos_qty = next((float(p['contracts']) for p in positions
                            if p['symbol'] == symbol and
                            p.get('side', '').upper() == side.upper()), 0)
            qty_close = round(pos_qty * fraction, 6)
            if qty_close <= 0:
                return {'status': 'nothing_to_close'}
            return self.exchange.create_order(symbol, 'market', close_side, qty_close,
                                              params={'reduceOnly': True})
        except Exception as e:
            print(f'[WEEX] close_partial failed: {e}')
            return {'error': str(e)}

    def modify_sl(self, order_id: str, new_sl_price: float,
                  symbol: str = '', side: str = '') -> Dict:
        """Update stop-loss on an open position. Paper: update in-memory. Live: place new stop order."""
        if self.paper:
            pos = next((p for p in self._paper_positions if p['id'] == order_id), None)
            if not pos:
                pos = next((p for p in self._paper_positions
                            if p['symbol'] == symbol and p['side'] == side), None)
            if not pos:
                return {'error': 'position_not_found'}
            old_sl = pos['sl']
            pos['sl'] = new_sl_price
            print(f'[PAPER] SL updated {pos["symbol"]} {old_sl} -> {new_sl_price}')
            return {'status': 'paper_sl_updated', 'old_sl': old_sl, 'new_sl': new_sl_price}
        try:
            stop_side = 'sell' if side == 'LONG' else 'buy'
            result = self.exchange.create_order(
                symbol, 'stop_market', stop_side, 0,
                params={'stopPrice': new_sl_price, 'reduceOnly': True, 'closePosition': True},
            )
            print(f'[WEEX] SL updated {symbol} → {new_sl_price}')
            return result
        except Exception as e:
            print(f'[WEEX] modify_sl failed: {e}')
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
