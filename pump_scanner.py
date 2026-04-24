"""
pump_scanner.py — Phase 4
Scans all USDT pairs for unusual volume/price spikes that may precede a
tradeable move. Results are sent to the dashboard via WebSocket so the
auto-scanner can prioritise these coins.
"""
import time
import ccxt
from typing import List, Dict


# Thresholds
_VOL_SPIKE_X   = 3.0   # volume must be Nx the 24h average to qualify
_PRICE_MOVE_PCT = 2.5  # price move % in last hour to qualify
_MIN_VOL_USDT  = 1_000_000  # minimum 24h volume to avoid micro-caps


class PumpScanner:
    def __init__(self, exchange: ccxt.Exchange):
        self.exchange = exchange
        self._cache: dict = {'results': [], 'ts': 0}

    # ── PUBLIC ────────────────────────────────────────────────────────────────

    def scan(self, force: bool = False) -> List[Dict]:
        """
        Returns list of coins showing unusual activity.
        Results cached 3 minutes (pump signals go stale fast).
        Set force=True to bypass cache.
        """
        now = time.time()
        if not force and now - self._cache['ts'] < 180 and self._cache['results']:
            return self._cache['results']

        results = []
        try:
            tickers = self.exchange.fetch_tickers()
            for symbol, t in tickers.items():
                if not symbol.endswith('/USDT'):
                    continue
                base = symbol.replace('/USDT', '')
                if not base.isascii() or not base.isalnum():
                    continue
                if base in ('USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDP', 'FDUSD'):
                    continue

                quote_vol   = t.get('quoteVolume') or 0
                pct_change  = t.get('percentage')  or 0
                last_price  = t.get('last')         or 0

                if quote_vol < _MIN_VOL_USDT or last_price <= 0:
                    continue

                # Volume spike: current vs baseVolume (24h avg approximation)
                base_vol  = t.get('baseVolume') or 0
                avg_vol   = (base_vol * last_price) if base_vol else 0

                vol_ratio = quote_vol / avg_vol if avg_vol > 0 else 0

                is_pump   = (abs(pct_change) >= _PRICE_MOVE_PCT
                             and (vol_ratio >= _VOL_SPIKE_X or quote_vol >= 10_000_000))

                if is_pump:
                    direction = 'UP' if pct_change > 0 else 'DOWN'
                    results.append({
                        'symbol':    symbol,
                        'price':     round(last_price, 6),
                        'change_pct': round(pct_change, 2),
                        'volume_usdt': round(quote_vol, 0),
                        'vol_ratio':  round(vol_ratio, 1),
                        'direction':  direction,
                        'score':      round(abs(pct_change) * (vol_ratio or 1), 2),
                    })

            # Sort by score descending — biggest moves with highest relative volume first
            results.sort(key=lambda x: x['score'], reverse=True)
            results = results[:20]

            self._cache = {'results': results, 'ts': now}
            print(f'[PumpScanner] Found {len(results)} pumping coins')

        except Exception as e:
            print(f'[PumpScanner] Error: {e}')

        return results

    def get_pump_symbols(self) -> List[str]:
        """Just the symbol list, for feeding into the main scanner."""
        return [r['symbol'] for r in self.scan()]

    def top_movers(self, n: int = 5) -> List[Dict]:
        return self.scan()[:n]
