import os
import requests

class WhatsAppNotifier:
    def __init__(self, phone: str = '', api_key: str = ''):
        self.instance_id = os.getenv('WA_INSTANCE_ID', '')
        self.api_token   = os.getenv('WA_API_TOKEN', '')
        # constructor args override env vars (main.py passes them, that's fine)
        self.phone   = phone or os.getenv('WA_PHONE_NUMBER', '')
        self.enabled = bool(self.instance_id and self.api_token and self.phone)

        if not self.enabled:
            print('[WhatsApp] Disabled — set WA_INSTANCE_ID, WA_API_TOKEN, WA_PHONE_NUMBER in .env')

    # ── Core sender ───────────────────────────────────────────────────────────
    def _send(self, message: str):
        if not self.enabled:
            return
        try:
            url = (f"https://api.green-api.com/waInstance{self.instance_id}"
                   f"/sendMessage/{self.api_token}")
            r = requests.post(url, json={
                "chatId":   f"{self.phone}@c.us",
                "message":  message
            }, timeout=10)
            if r.status_code != 200:
                print(f'[WhatsApp] Error {r.status_code}: {r.text}')
        except Exception as e:
            print(f'[WhatsApp] Exception: {e}')

    # ── Signal alert ──────────────────────────────────────────────────────────
    def send_signal(self, signal: dict):
        action = signal.get('action', '?')
        symbol = signal.get('symbol', '?')
        emoji  = '🟢' if action == 'LONG' else '🔴' if action == 'SHORT' else '⏸️'
        self._send(
            f"{emoji} SIGNAL: {action} {symbol}\n"
            f"Entry: {signal.get('entry')} | Conf: {signal.get('confidence')}%\n"
            f"SL: {signal.get('sl')} | TP1: {signal.get('tp1')}\n"
            f"R:R {signal.get('rr','?')} | Setup: {signal.get('setup_type','?')}\n"
            f"Veto: {signal.get('veto_decision','APPROVE')}\n"
            f"💡 {str(signal.get('reason',''))[:120]}"
        )

    # ── Trade opened — two call signatures, both supported ────────────────────
    def send_trade_opened(self, symbol: str, action: str, qty, entry, sl, tp1):
        emoji = '🟢' if action == 'LONG' else '🔴'
        self._send(
            f"✅ TRADE OPENED\n"
            f"{emoji} {action} {symbol}\n"
            f"Entry: {entry} | Qty: {qty}\n"
            f"SL: {sl} | TP1: {tp1}"
        )

    def notify_trade_opened(self, order: dict, signal: dict):
        """Called as notify_trade_opened(result, signal) — delegates to send_trade_opened."""
        self.send_trade_opened(
            symbol = signal.get('symbol', '?'),
            action = signal.get('action', '?'),
            qty    = order.get('qty', '?'),
            entry  = signal.get('entry'),
            sl     = signal.get('sl'),
            tp1    = signal.get('tp1'),
        )

    # ── Scan summary ──────────────────────────────────────────────────────────
    def send_scan_summary(self, total_scanned: int, signals_found: int, next_in_sec: int):
        self._send(
            f"📊 SCAN DONE\n"
            f"Scanned: {total_scanned} coins\n"
            f"Signals: {signals_found}\n"
            f"Next in {next_in_sec // 60} min"
        )

    # ── Risk / bot alerts ─────────────────────────────────────────────────────
    def bot_paused(self, reason: str, minutes: int = 0):
        self._send(f"⏸️ BOT PAUSED ({minutes} min)\n{reason}")

    # ── Position events ───────────────────────────────────────────────────────
    def notify_position_event(self, ev: dict):
        event = ev.get('event', '?').upper()
        emoji = '💰' if 'TP' in event else '🛑' if 'SL' in event else '📊'
        self._send(
            f"{emoji} {event} — {ev.get('symbol','?')}\n"
            f"PnL: {ev.get('pnl', 0):+.2f} USDT"
        )

    def notify_emergency_stop(self, closed_count: int):
        self._send(f"🚨 EMERGENCY STOP\n{closed_count} position(s) closed!")

    def trade_closed(self, symbol: str, side: str, pnl: float, reason: str):
        emoji = '💰' if pnl > 0 else '🛑'
        self._send(
            f"{emoji} TRADE CLOSED — {side} {symbol}\n"
            f"Reason: {reason}\n"
            f"PnL: {pnl:+.2f} USDT"
        )

    def notify_daily_limit(self):
        self._send("⛔ DAILY LOSS LIMIT HIT — Trading paused for today.")