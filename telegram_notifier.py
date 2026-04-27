import requests
import time
from concurrent.futures import ThreadPoolExecutor


class TelegramNotifier:
    """
    Sends trading signal alerts to a Telegram chat.
    Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env to enable.
    """

    def __init__(self, token: str, chat_id: str):
        self.token    = token.strip() if token else ''
        self.chat_id  = chat_id.strip() if chat_id else ''
        self._base    = f'https://api.telegram.org/bot{self.token}'
        self._cooldown: dict[str, float] = {}
        # Background thread pool — sends never block the scanner loop
        self._pool    = ThreadPoolExecutor(max_workers=2, thread_name_prefix='tg')

    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    # ── public helpers ────────────────────────────────────────────────────────

    def send_signal(self, signal: dict) -> bool:
        action = signal.get('action', 'WAIT')
        symbol = signal.get('symbol', '')

        if action not in ('LONG', 'SHORT'):
            return False

        # 5-min cooldown per symbol+direction to avoid spam
        key = f'{symbol}:{action}'
        if time.time() - self._cooldown.get(key, 0) < 300:
            return False
        self._cooldown[key] = time.time()

        emoji = '🟢' if action == 'LONG' else '🔴'
        conf  = signal.get('confidence', 0)
        bar   = '█' * (conf // 10) + '░' * (10 - conf // 10)

        entry       = signal.get('entry', 0)
        sl          = signal.get('sl', 0)
        tp1         = signal.get('tp1', 0)
        tp2         = signal.get('tp2', 0)
        rr          = signal.get('rr', 'N/A')
        setup       = signal.get('setup_type', 'N/A')
        pattern     = signal.get('candle_pattern', 'N/A')
        session     = signal.get('session', 'N/A')
        sentiment   = signal.get('sentiment', 'N/A')
        key_level   = signal.get('key_level', 'N/A')
        invalidation = signal.get('invalidation', 'N/A')
        reason      = (signal.get('reason') or '')[:220]

        text = (
            f'{emoji} *{action} — {symbol}*\n'
            f'`{bar}` {conf}/100\n'
            f'━━━━━━━━━━━━━━━━━━━━\n'
            f'💰 Entry: `{entry:,.4f}`\n'
            f'🛑 SL:    `{sl:,.4f}`\n'
            f'✅ TP1:   `{tp1:,.4f}`\n'
            f'✅ TP2:   `{tp2:,.4f}`\n'
            f'📊 R:R    `{rr}`\n'
            f'━━━━━━━━━━━━━━━━━━━━\n'
            f'🔍 Setup:     {setup}\n'
            f'🕯 Pattern:   {pattern}\n'
            f'⏰ Session:   {session}\n'
            f'😱 Sentiment: {sentiment}\n'
            f'━━━━━━━━━━━━━━━━━━━━\n'
            f'📌 Key level: `{key_level}`\n'
            f'❌ Invalidate: `{invalidation}`\n'
            f'━━━━━━━━━━━━━━━━━━━━\n'
            f'_{reason}_'
        )
        return self._send(text)

    def send_trade_opened(self, symbol: str, action: str,
                          qty: float, entry: float, sl: float, tp1: float) -> bool:
        emoji = '🟢' if action == 'LONG' else '🔴'
        text = (
            f'{emoji} *TRADE OPENED — {symbol}*\n'
            f'Direction: `{action}`  |  Qty: `{qty}`\n'
            f'Entry: `{entry:,.4f}`\n'
            f'SL:    `{sl:,.4f}`\n'
            f'TP1:   `{tp1:,.4f}`\n'
            f'⚠️ Monitor your position!'
        )
        return self._send(text)

    def send_position_event(self, event: dict) -> bool:
        ev     = event.get('event', '')
        symbol = event.get('symbol', '')
        pnl    = event.get('pnl', 0)
        emoji  = '✅' if pnl >= 0 else '❌'
        text = (
            f'{emoji} *{ev.upper()} — {symbol}*\n'
            f'PnL: `${pnl:+.2f}`'
        )
        return self._send(text)

    def send_scan_summary(self, total: int, signals: int, next_in_sec: int) -> bool:
        if signals == 0:
            return False   # don't spam with "0 signals found"
        text = (
            f'📡 *Scan done* — {signals} signal(s) found out of {total} coins\n'
            f'Next scan in {next_in_sec // 60} min'
        )
        return self._send(text)

    def send_error(self, msg: str) -> bool:
        return self._send(f'⚠️ *Bot error*\n`{msg[:300]}`')

    # ── named aliases used by TradeManager / main ────────────────────────────

    def signal_generated(self, signal: dict) -> bool:
        return self.send_signal(signal)

    def trade_opened(self, symbol: str, side: str, qty: float,
                     entry: float, sl: float, tp1: float) -> bool:
        return self.send_trade_opened(symbol, side, qty, entry, sl, tp1)

    def trade_closed(self, symbol: str, side: str, pnl: float, reason: str) -> bool:
        emoji = '✅' if pnl >= 0 else '❌'
        text = (
            f'{emoji} *TRADE CLOSED — {symbol}*\n'
            f'Direction: `{side}`  |  Reason: `{reason}`\n'
            f'PnL: `${pnl:+.2f} USDT`'
        )
        return self._send(text)

    def bot_paused(self, reason: str, resume_in_min: int) -> bool:
        text = (
            f'⏸ *BOT PAUSED*\n'
            f'Reason: {reason}\n'
            f'Resumes in: {resume_in_min} minutes'
        )
        return self._send(text)

    def daily_summary(self, stats: dict) -> bool:
        pnl    = stats.get('total_pnl', 0)
        emoji  = '🟢' if pnl >= 0 else '🔴'
        text = (
            f'{emoji} *Daily Summary*\n'
            f'Trades: `{stats.get("trades_taken", 0)}` '
            f'(W:{stats.get("wins", 0)} L:{stats.get("losses", 0)})\n'
            f'Win Rate: `{stats.get("win_rate", 0)}%`\n'
            f'PnL: `${pnl:+.2f} USDT`\n'
            f'Drawdown: `{stats.get("drawdown_pct", 0):.1f}%`'
        )
        return self._send(text)

    def error(self, msg: str) -> bool:
        return self.send_error(msg)

    # ── private ───────────────────────────────────────────────────────────────

    def _send(self, text: str) -> bool:
        """Fire-and-forget: submits to background thread, returns immediately."""
        if not self.enabled():
            return False
        self._pool.submit(self._send_sync, text)
        return True

    def _send_sync(self, text: str) -> bool:
        """Actual blocking HTTP call — runs in a background thread."""
        try:
            r = requests.post(
                f'{self._base}/sendMessage',
                json={
                    'chat_id':                  self.chat_id,
                    'text':                     text,
                    'parse_mode':               'Markdown',
                    'disable_web_page_preview': True,
                },
                timeout=5,   # reduced from 10 — fail fast
            )
            if not r.ok:
                print(f'[Telegram] {r.status_code}: {r.text[:200]}')
            return r.ok
        except Exception as e:
            # Log short form — avoid flooding console with full stack trace
            print(f'[Telegram] Send failed: {type(e).__name__}: {str(e)[:120]}')
            return False
