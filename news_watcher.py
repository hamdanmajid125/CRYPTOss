"""
news_watcher.py — Phase 3
Fetches upcoming high-impact macro events from ForexFactory RSS and enforces
a configurable blackout window around them (default ±30 min).
Bot checks this before placing any trade.
"""
import time
import datetime
import xml.etree.ElementTree as ET
import requests


# High-impact keywords that affect crypto markets
_IMPACT_KEYWORDS = (
    'CPI', 'PPI', 'FOMC', 'Fed', 'Interest Rate', 'NFP', 'Non-Farm',
    'GDP', 'Unemployment', 'Retail Sales', 'PCE', 'Powell', 'Inflation',
    'Treasury', 'SEC', 'Crypto', 'Bitcoin', 'ETF',
)

_FF_RSS = 'https://nfs.faireconomy.media/ff_calendar_thisweek.xml'
_CACHE_TTL = 3600  # refresh event list every hour


class NewsWatcher:
    def __init__(self, blackout_minutes: int = 30):
        self.blackout_sec = blackout_minutes * 60
        self._events: list = []
        self._last_fetch: float = 0

    # ── PUBLIC ────────────────────────────────────────────────────────────────

    def is_blackout(self) -> tuple:
        """
        Returns (True, reason_str) if we are within the blackout window of a
        high-impact event, otherwise (False, '').
        """
        self._refresh()
        now = time.time()
        for ev in self._events:
            diff = abs(ev['ts'] - now)
            if diff <= self.blackout_sec:
                direction = 'in' if ev['ts'] > now else 'ago'
                mins = int(diff / 60)
                return True, (f"Blackout: {ev['title']} "
                              f"({mins} min {direction}) — no new trades")
        return False, ''

    def next_event(self) -> dict:
        """Returns the next upcoming high-impact event or empty dict."""
        self._refresh()
        now = time.time()
        upcoming = [e for e in self._events if e['ts'] > now]
        return upcoming[0] if upcoming else {}

    def get_events(self) -> list:
        self._refresh()
        return list(self._events)

    # ── PRIVATE ───────────────────────────────────────────────────────────────

    def _refresh(self):
        if time.time() - self._last_fetch < _CACHE_TTL:
            return
        try:
            r = requests.get(_FF_RSS, timeout=8,
                             headers={'User-Agent': 'Mozilla/5.0'})
            if not r.ok:
                return
            root = ET.fromstring(r.content)
            events = []
            for item in root.iter('item'):
                title    = (item.findtext('title') or '').strip()
                country  = (item.findtext('country') or '').upper()
                impact   = (item.findtext('impact') or '').lower()
                pub_date = item.findtext('pubDate') or ''

                if country not in ('USD', 'US') and 'crypto' not in title.lower():
                    continue
                if impact not in ('high', 'medium'):
                    continue
                if not any(k.lower() in title.lower() for k in _IMPACT_KEYWORDS):
                    continue

                ts = self._parse_date(pub_date)
                if ts:
                    events.append({'title': title, 'ts': ts, 'impact': impact})

            events.sort(key=lambda e: e['ts'])
            self._events = events
            self._last_fetch = time.time()
            print(f'[NewsWatcher] Loaded {len(events)} high-impact events')
        except Exception as e:
            print(f'[NewsWatcher] Fetch error: {e}')

    @staticmethod
    def _parse_date(date_str: str) -> float:
        formats = (
            '%a, %d %b %Y %H:%M:%S %z',
            '%a, %d %b %Y %H:%M:%S GMT',
        )
        for fmt in formats:
            try:
                dt = datetime.datetime.strptime(date_str.strip(), fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=datetime.timezone.utc)
                return dt.timestamp()
            except ValueError:
                continue
        return 0.0
