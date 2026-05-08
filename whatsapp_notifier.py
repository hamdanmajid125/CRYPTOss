import time
import urllib.parse
import requests
from concurrent.futures import ThreadPoolExecutor

_API_URL = 'https://api.callmebot.com/whatsapp.php'


class WhatsAppNotifier:
    """
    Sends trading alerts via WhatsApp using the free CallMeBot service.

    Setup (one-time on your phone):
      1. Add +34 644 01 99 57 to contacts.
      2. Send: I allow callmebot to send me messages
      3. You receive your apikey by reply.

    .env keys required:
      WHATSAPP_PHONE   = 923xxxxxxxxx  (international format, no +)
      WHATSAPP_API_KEY = xxxx
    """

    def __init__(self, phone: str, api_key: str):
        self.phone   = phone.strip() if phone else ''
        self.api_key = api_key.strip() if api_key else ''
        self._cooldown: dict[str, float] = {}
        self._pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix='wa')

    def enabled(self) -> bool:
        return bool(self.phone and self.api_key)

    # ── public helpers ────────────────────────────────────────────────────────

    def send_signal(self, signal: dict) -> bool:
        action = signal.get('action', 'WAIT')
        symbol = signal.get('symbol', '')

        if action not in ('LONG', 'SHORT'):
            return False

        key = f'{symbol}:{action}'
        if time.time() - self._cooldown.get(key, 0) < 300:
            return False
        self._cooldown[key] = time.time()

        arrow  = '▲' if action == 'LONG' else '▼'
        conf   = signal.get('confidence', 0)
        bar    = '█' * (conf // 10) + '░' * (10 - conf // 10)
        entry  = signal.get('entry', 0)
        sl     = signal.get('sl', 0)
        tp1    = signal.get('tp1', 0)
        tp2    = signal.get('tp2', 0)
        rr     = signal.get('rr', 'N/A')
        reason = (signal.get('reason') or '')[:200]

        text = (
            f'{arrow} {action} - {symbol}\n'
            f'{bar} {conf}/100\n'
            f'Entry: {entry:,.4f}\n'
            f'SL:    {sl:,.4f}\n'
            f'TP1:   {tp1:,.4f}\n'
            f'TP2:   {tp2:,.4f}\n'
            f'R:R    {rr}\n'
            f'{reason}'
        )
        return self._send(text)

    def send_trade_opened(self, symbol: str, action: str,
                          qty: float, entry: float, sl: float, tp1: float) -> bool:
        arrow = '▲' if action == 'LONG' else '▼'
        text = (
            f'{arrow} TRADE OPENED - {symbol}\n'
            f'Direction: {action}  Qty: {qty}\n'
            f'Entry: {entry:,.4f}\n'
            f'SL:    {sl:,.4f}\n'
            f'TP1:   {tp1:,.4f}'
        )
        return self._send(text)

    def send_position_event(self, event: dict) -> bool:
        ev     = event.get('event', '')
        symbol = event.get('symbol', '')
        pnl    = event.get('pnl', 0)
        sign   = '+' if pnl >= 0 else ''
        text   = f'{"[WIN]" if pnl >= 0 else "[LOSS]"} {ev.upper()} - {symbol}\nPnL: ${sign}{pnl:.2f}'
        return self._send(text)

    def send_scan_summary(self, total: int, signals: int, next_in_sec: int) -> bool:
        if signals == 0:
            return False
        text = f'[SCAN] {signals} signal(s) from {total} coins. Next in {next_in_sec // 60} min.'
        return self._send(text)

    def send_error(self, msg: str) -> bool:
        return self._send(f'[ERROR] {msg[:300]}')

    # ── named aliases (same interface as TelegramNotifier) ───────────────────

    def signal_generated(self, signal: dict) -> bool:
        return self.send_signal(signal)

    def trade_opened(self, symbol: str, side: str, qty: float,
                     entry: float, sl: float, tp1: float) -> bool:
        return self.send_trade_opened(symbol, side, qty, entry, sl, tp1)

    def trade_closed(self, symbol: str, side: str, pnl: float, reason: str) -> bool:
        sign = '+' if pnl >= 0 else ''
        text = (
            f'{"[WIN]" if pnl >= 0 else "[LOSS]"} TRADE CLOSED - {symbol}\n'
            f'Direction: {side}  Reason: {reason}\n'
            f'PnL: ${sign}{pnl:.2f} USDT'
        )
        return self._send(text)

    def bot_paused(self, reason: str, resume_in_min: int) -> bool:
        return self._send(f'[PAUSED] {reason}\nResumes in: {resume_in_min} min')

    def daily_summary(self, stats: dict) -> bool:
        pnl  = stats.get('total_pnl', 0)
        sign = '+' if pnl >= 0 else ''
        text = (
            f'[DAILY] Trades: {stats.get("trades_taken", 0)} '
            f'(W:{stats.get("wins", 0)} L:{stats.get("losses", 0)})\n'
            f'Win Rate: {stats.get("win_rate", 0)}%\n'
            f'PnL: ${sign}{pnl:.2f} USDT  DD: {stats.get("drawdown_pct", 0):.1f}%'
        )
        return self._send(text)

    def error(self, msg: str) -> bool:
        return self.send_error(msg)

    # ── private ───────────────────────────────────────────────────────────────

    def _send(self, text: str) -> bool:
        if not self.enabled():
            return False
        self._pool.submit(self._send_sync, text)
        return True

    def _send_sync(self, text: str) -> bool:
        try:
            r = requests.get(
                _API_URL,
                params={
                    'phone':  self.phone,
                    'text':   urllib.parse.quote(text),
                    'apikey': self.api_key,
                },
                timeout=10,
            )
            if not r.ok:
                print(f'[WhatsApp] {r.status_code}: {r.text[:200]}')
            return r.ok
        except Exception as e:
            print(f'[WhatsApp] Send failed: {type(e).__name__}: {str(e)[:120]}')
            return False
