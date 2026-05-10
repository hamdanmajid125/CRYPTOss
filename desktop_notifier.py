"""
Windows desktop toast notifications via winotify.
Runs sends in a daemon thread so the bot loop is never blocked.
"""
import threading
from winotify import Notification, audio

APP_ID = 'Crypto Trading Bot'


def _toast(title: str, body: str, duration: str = 'short') -> None:
    """Show a Windows toast. Called from a background thread."""
    try:
        n = Notification(
            app_id=APP_ID,
            title=title,
            msg=body[:200],
            duration=duration,
        )
        n.set_audio(audio.Default, loop=False)
        n.show()
    except Exception as e:
        print(f'[Desktop] Notification failed: {e}')


def notify(title: str, body: str, duration: str = 'short') -> None:
    """Fire-and-forget desktop toast (non-blocking)."""
    threading.Thread(target=_toast, args=(title, body, duration),
                     daemon=True).start()


# ── Typed helpers (same events as WhatsApp) ───────────────────────────────────

def notify_signal(signal: dict) -> None:
    action = signal.get('action', '')
    if action not in ('LONG', 'SHORT'):
        return
    symbol = signal.get('symbol', '')
    conf   = signal.get('confidence', 0)
    entry  = signal.get('entry', 0)
    rr     = signal.get('rr', '?')
    arrow  = '▲' if action == 'LONG' else '▼'
    notify(
        title=f'{arrow} {action}  {symbol}',
        body=f'Entry {entry:,.4f}  |  Conf {conf}%  |  R:R {rr}',
    )


def notify_trade_opened(symbol: str, action: str, entry: float, sl: float, tp1: float) -> None:
    arrow = '▲' if action == 'LONG' else '▼'
    notify(
        title=f'Trade Opened  {arrow} {symbol}',
        body=f'Entry {entry:,.4f}  SL {sl:,.4f}  TP1 {tp1:,.4f}',
    )


def notify_trade_closed(symbol: str, action: str, pnl: float, reason: str) -> None:
    sign = '+' if pnl >= 0 else ''
    label = 'WIN' if pnl >= 0 else 'LOSS'
    notify(
        title=f'[{label}] Trade Closed  {symbol}',
        body=f'{action}  PnL {sign}{pnl:.2f} USDT  ({reason})',
    )


def notify_risk_alert(event: str, reason: str, minutes: int = 0) -> None:
    titles = {
        'bot_paused':       f'Bot Paused ({minutes} min)',
        'daily_loss_limit': 'Daily Loss Limit Hit',
        'drawdown_halt':    'Drawdown Halt Triggered',
    }
    notify(
        title=titles.get(event, f'Risk Alert: {event}'),
        body=reason[:200],
        duration='long',
    )
