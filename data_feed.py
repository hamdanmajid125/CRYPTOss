import ccxt
import pandas as pd
import pandas_ta_classic as ta
import requests
import time
import xml.etree.ElementTree as ET
from utils import with_retry

class DataFeed:

    def __init__(self, exchange: ccxt.Exchange, cryptopanic_token: str = ''):
        self.exchange = exchange
        self._fng_cache = {'value': 50, 'label': 'Neutral', 'ts': 0}
        self._top_coins_cache = {'coins': [], 'ts': 0}
        self._news_cache: dict = {}          # symbol → {sentiment, ts}
        self._cryptopanic_token = cryptopanic_token

    # ─────────────────────────────────────────────────────────────────────
    # TOP COINS BY VOLUME  (NEW - replaces hardcoded SYMBOLS in .env)
    # ─────────────────────────────────────────────────────────────────────
    def get_top_coins_by_volume(self, top_n: int = 20, min_volume_usdt: float = 5_000_000) -> list:
        """
        Fetches all USDT spot markets from Binance public API,
        ranks by 24h quote volume, returns top_n symbols.
        Results are cached for 10 minutes to avoid hammering the API.
        """
        now = time.time()
        if now - self._top_coins_cache['ts'] < 600 and self._top_coins_cache['coins']:
            return self._top_coins_cache['coins']

        try:
            tickers = self.exchange.fetch_tickers()
            usdt_pairs = []
            for symbol, t in tickers.items():
                if not symbol.endswith('/USDT'):
                    continue
                # Skip stablecoins and wrapped tokens
                base = symbol.replace('/USDT', '')
                if base in ('USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'USDP', 'FDUSD'):
                    continue
                # Skip coins with non-ASCII characters (e.g. Chinese-name tokens)
                if not base.isascii() or not base.isalnum():
                    continue
                quote_vol = t.get('quoteVolume') or 0
                if quote_vol < min_volume_usdt:
                    continue
                usdt_pairs.append({
                    'symbol': symbol,
                    'volume': quote_vol,
                    'change': t.get('percentage', 0) or 0,
                    'price': t.get('last', 0) or 0,
                })

            # Sort by volume descending, pick top_n
            usdt_pairs.sort(key=lambda x: x['volume'], reverse=True)
            top = [p['symbol'] for p in usdt_pairs[:top_n]]
            self._top_coins_cache = {'coins': top, 'ts': now}
            print(f'[DataFeed] Top {top_n} coins by volume: {top}')
            return top

        except Exception as e:
            print(f'[DataFeed] Error fetching top coins: {e}')
            # Fallback to known liquid coins
            return [
                'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
                'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT',
                'MATIC/USDT', 'UNI/USDT', 'LTC/USDT', 'ATOM/USDT', 'NEAR/USDT',
                'FIL/USDT', 'INJ/USDT', 'ARB/USDT', 'OP/USDT', 'SUI/USDT',
            ]

    # ─────────────────────────────────────────────────────────────────────
    # OHLCV + INDICATORS
    # ─────────────────────────────────────────────────────────────────────
    def get_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 350) -> pd.DataFrame:
        raw = with_retry(lambda: self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit))
        df = pd.DataFrame(raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        # Force all price/volume columns to float64 — exchanges can return None
        # for thinly-traded or newly-listed tokens, which would stay as Python
        # None in an object-dtype column and break arithmetic downstream.
        for col in ('open', 'high', 'low', 'close', 'volume'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['ema20']  = ta.ema(df['close'], length=20)
        df['ema50']  = ta.ema(df['close'], length=50)
        ema200       = ta.ema(df['close'], length=200)
        df['ema200'] = ema200 if ema200 is not None else float('nan')
        df['rsi']    = ta.rsi(df['close'], length=14)
        df['atr']    = ta.atr(df['high'], df['low'], df['close'], length=14)

        macd = ta.macd(df['close'])
        df['macd']     = macd['MACD_12_26_9']
        df['macd_sig'] = macd['MACDs_12_26_9']

        bb = ta.bbands(df['close'])
        bb_upper_col = [c for c in bb.columns if c.startswith('BBU')][0]
        bb_lower_col = [c for c in bb.columns if c.startswith('BBL')][0]
        df['bb_upper'] = bb[bb_upper_col]
        df['bb_lower'] = bb[bb_lower_col]

        df['volume_sma'] = ta.sma(df['volume'], length=20)
        return df

    # ─────────────────────────────────────────────────────────────────────
    # CANDLESTICK PATTERN DETECTION  (NEW)
    # ─────────────────────────────────────────────────────────────────────
    def detect_candle_pattern(self, df: pd.DataFrame) -> str:
        """
        Detects the most recent significant candlestick pattern.
        Checks last 3 candles for: Hammer, Shooting Star, Engulfing, Doji, Morning Star, Evening Star.
        Returns pattern name or 'No pattern'.
        """
        if len(df) < 3:
            return 'No pattern'

        c  = df.iloc[-1]   # current candle
        p  = df.iloc[-2]   # previous candle
        p2 = df.iloc[-3]   # 2 candles ago

        body      = abs(c['close'] - c['open'])
        full_range = c['high'] - c['low']
        if full_range == 0:
            return 'No pattern'

        upper_wick = c['high'] - max(c['close'], c['open'])
        lower_wick = min(c['close'], c['open']) - c['low']
        body_pct   = body / full_range

        p_body      = abs(p['close'] - p['open'])
        p_full_range = p['high'] - p['low']

        # Doji: body < 10% of range
        if body_pct < 0.10:
            return 'Doji — indecision, wait for next candle confirmation'

        # Hammer (bullish reversal): lower wick >= 2x body, small upper wick, near lows
        if (lower_wick >= 2 * body and upper_wick <= body * 0.3
                and c['close'] < df['close'].tail(20).mean()):
            return 'Hammer — bullish reversal signal at support'

        # Shooting Star (bearish reversal): upper wick >= 2x body, small lower wick, near highs
        if (upper_wick >= 2 * body and lower_wick <= body * 0.3
                and c['close'] > df['close'].tail(20).mean()):
            return 'Shooting Star — bearish reversal signal at resistance'

        # Bullish Engulfing: previous bearish, current bullish and larger
        if (p['close'] < p['open'] and c['close'] > c['open']
                and c['open'] < p['close'] and c['close'] > p['open']
                and body > p_body):
            return 'Bullish Engulfing — strong LONG signal, high confidence'

        # Bearish Engulfing: previous bullish, current bearish and larger
        if (p['close'] > p['open'] and c['close'] < c['open']
                and c['open'] > p['close'] and c['close'] < p['open']
                and body > p_body):
            return 'Bearish Engulfing — strong SHORT signal, high confidence'

        # Bullish Marubozu: large bullish candle, tiny wicks
        if (c['close'] > c['open'] and body_pct > 0.85
                and upper_wick < body * 0.1 and lower_wick < body * 0.1):
            return 'Bullish Marubozu — strong buying pressure, momentum LONG'

        # Bearish Marubozu: large bearish candle, tiny wicks
        if (c['close'] < c['open'] and body_pct > 0.85
                and upper_wick < body * 0.1 and lower_wick < body * 0.1):
            return 'Bearish Marubozu — strong selling pressure, momentum SHORT'

        # Morning Star (3-candle bullish reversal)
        if (p2['close'] < p2['open']           # bearish candle
                and abs(p['close'] - p['open']) < p_full_range * 0.3  # small body star
                and c['close'] > c['open']      # bullish close
                and c['close'] > (p2['open'] + p2['close']) / 2):
            return 'Morning Star — 3-candle bullish reversal, strong LONG setup'

        # Evening Star (3-candle bearish reversal)
        if (p2['close'] > p2['open']
                and abs(p['close'] - p['open']) < p_full_range * 0.3
                and c['close'] < c['open']
                and c['close'] < (p2['open'] + p2['close']) / 2):
            return 'Evening Star — 3-candle bearish reversal, strong SHORT setup'

        return 'No significant pattern — use SMC confluence only'

    # ─────────────────────────────────────────────────────────────────────
    # VOLUME
    # ─────────────────────────────────────────────────────────────────────
    def get_volume_signal(self, df: pd.DataFrame) -> str:
        last = df.iloc[-1]
        if pd.isna(last['volume_sma']) or last['volume_sma'] == 0:
            return 'NORMAL'
        ratio = last['volume'] / last['volume_sma']
        if ratio >= 2.0:
            return 'VERY HIGH — strong confirmation'
        elif ratio >= 1.5:
            return 'HIGH — confirms direction'
        elif ratio <= 0.5:
            return 'VERY LOW — weak, avoid trading'
        elif ratio <= 0.7:
            return 'LOW — signal is weaker'
        return 'NORMAL'

    def get_volume_trend(self, df: pd.DataFrame) -> str:
        """Is volume increasing or decreasing over last 5 candles?"""
        recent_vols = df['volume'].tail(5).values
        if recent_vols[-1] > recent_vols[0] * 1.3:
            return 'INCREASING — momentum building'
        elif recent_vols[-1] < recent_vols[0] * 0.7:
            return 'DECREASING — momentum fading'
        return 'FLAT'

    # ─────────────────────────────────────────────────────────────────────
    # RSI DIVERGENCE
    # ─────────────────────────────────────────────────────────────────────
    def detect_rsi_divergence(self, df: pd.DataFrame) -> str:
        if len(df) < 25:
            return 'NONE'
        recent_window = df.tail(5)
        prev_window = df.iloc[-20:-5]
        if recent_window.empty or prev_window.empty:
            return 'NONE'
        try:
            recent_high_idx = recent_window['close'].idxmax()
            prev_high_idx = prev_window['close'].idxmax()
            recent_low_idx = recent_window['close'].idxmin()
            prev_low_idx = prev_window['close'].idxmin()

            recent_high_price = recent_window.loc[recent_high_idx, 'close']
            prev_high_price = prev_window.loc[prev_high_idx, 'close']
            recent_high_rsi = recent_window.loc[recent_high_idx, 'rsi']
            prev_high_rsi = prev_window.loc[prev_high_idx, 'rsi']

            recent_low_price = recent_window.loc[recent_low_idx, 'close']
            prev_low_price = prev_window.loc[prev_low_idx, 'close']
            recent_low_rsi = recent_window.loc[recent_low_idx, 'rsi']
            prev_low_rsi = prev_window.loc[prev_low_idx, 'rsi']

            if recent_high_price > prev_high_price and recent_high_rsi < prev_high_rsi:
                return 'BEARISH'
            if recent_low_price < prev_low_price and recent_low_rsi > prev_low_rsi:
                return 'BULLISH'
        except Exception:
            pass
        return 'NONE'

    # ─────────────────────────────────────────────────────────────────────
    # SMC
    # ─────────────────────────────────────────────────────────────────────
    def find_liquidity_levels(self, df: pd.DataFrame) -> dict:
        recent = df.tail(30)
        bsl = recent['high'].nlargest(3).tolist()
        ssl = recent['low'].nsmallest(3).tolist()
        return {'bsl': bsl, 'ssl': ssl}

    def find_order_blocks(self, df: pd.DataFrame) -> list:
        obs = []
        for i in range(5, len(df) - 3):
            candle = df.iloc[i]
            body_size = abs(candle['close'] - candle['open']) / candle['open']
            if body_size > 0.004:
                next3 = df.iloc[i + 1:i + 4]
                move = (next3['close'].iloc[-1] - candle['close']) / candle['close']
                if move > 0.008:
                    # Bullish OB = last BEARISH candle before upward move (demand zone)
                    # Zone: candle low → candle high; key entry level = candle open (top of body)
                    if candle['close'] < candle['open']:
                        obs.append({
                            'price': float(candle['open']),   # top of bearish body = entry trigger
                            'top':   float(candle['high']),
                            'bot':   float(candle['low']),
                            'type':  'bullish',
                            'time':  str(df.index[i]),
                        })
                elif move < -0.008:
                    # Bearish OB = last BULLISH candle before downward move (supply zone)
                    # Zone: candle low → candle high; key entry level = candle open (bottom of body)
                    if candle['close'] > candle['open']:
                        obs.append({
                            'price': float(candle['open']),   # bottom of bullish body = entry trigger
                            'top':   float(candle['high']),
                            'bot':   float(candle['low']),
                            'type':  'bearish',
                            'time':  str(df.index[i]),
                        })
        return obs[-3:]

    def find_fvgs(self, df: pd.DataFrame) -> list:
        """Detect Fair Value Gaps with fill and partial-fill detection."""
        fvgs = []
        for i in range(1, len(df) - 1):
            prev = df.iloc[i - 1]
            nxt  = df.iloc[i + 1]
            subsequent = df.iloc[i + 2:]

            if nxt['low'] > prev['high']:
                top      = float(nxt['low'])
                bot      = float(prev['high'])
                gap_size = top - bot
                filled   = False
                max_fill = 0.0
                for _, sc in subsequent.iterrows():
                    # Bull FVG filled when any wick enters the gap (low, not close)
                    if sc['low'] <= bot:
                        filled = True
                        break
                    fill_down = top - max(float(sc['low']), bot)
                    max_fill  = max(max_fill, min(fill_down / gap_size * 100, 100) if gap_size > 0 else 0)
                if not filled:
                    fvgs.append({'type': 'bull', 'top': top, 'bot': bot,
                                 'filled': False, 'partial_fill_pct': round(max_fill, 1)})

            if nxt['high'] < prev['low']:
                top      = float(prev['low'])
                bot      = float(nxt['high'])
                gap_size = top - bot
                filled   = False
                max_fill = 0.0
                for _, sc in subsequent.iterrows():
                    # Bear FVG filled when any wick enters the gap (high, not close)
                    if sc['high'] >= top:
                        filled = True
                        break
                    fill_up  = min(float(sc['high']), top) - bot
                    max_fill = max(max_fill, min(fill_up / gap_size * 100, 100) if gap_size > 0 else 0)
                if not filled:
                    fvgs.append({'type': 'bear', 'top': top, 'bot': bot,
                                 'filled': False, 'partial_fill_pct': round(max_fill, 1)})

        return fvgs[-4:]

    def market_structure(self, df: pd.DataFrame) -> str:
        last = df.iloc[-1]
        if last['close'] > last['ema20'] > last['ema50']:
            return 'Bullish'
        elif last['close'] < last['ema20'] < last['ema50']:
            return 'Bearish'
        return 'Ranging'

    # ─────────────────────────────────────────────────────────────────────
    # BOS / CHoCH DETECTION  (Phase 2A)
    # ─────────────────────────────────────────────────────────────────────
    def detect_bos_choch(self, df: pd.DataFrame) -> dict:
        """
        Detects the most recent Break of Structure (BOS) or Change of Character (CHoCH).
        Uses swing highs/lows over the last 50 candles.
        Returns dict with keys: type ('BOS_BULLISH'|'BOS_BEARISH'|'CHoCH_BULLISH'|'CHoCH_BEARISH'|'NONE'),
                                 level (price of the broken structure), confirmed (bool).
        """
        if len(df) < 20:
            return {'type': 'NONE', 'level': 0, 'confirmed': False}

        window = df.tail(50).copy()
        highs  = window['high'].values
        lows   = window['low'].values
        closes = window['close'].values

        def swing_highs(arr, lookback=5):
            result = []
            for i in range(lookback, len(arr) - lookback):
                if arr[i] == max(arr[i - lookback:i + lookback + 1]):
                    result.append((i, arr[i]))
            return result

        def swing_lows(arr, lookback=5):
            result = []
            for i in range(lookback, len(arr) - lookback):
                if arr[i] == min(arr[i - lookback:i + lookback + 1]):
                    result.append((i, arr[i]))
            return result

        sh = swing_highs(highs)
        sl = swing_lows(lows)

        if len(sh) < 2 or len(sl) < 2:
            return {'type': 'NONE', 'level': 0, 'confirmed': False}

        last_close = closes[-1]
        prev_close = closes[-2]

        # BOS Bullish: close breaks above the most recent swing high
        _, last_sh_price = sh[-1]
        _, prev_sh_price = sh[-2]

        # BOS Bearish: close breaks below the most recent swing low
        _, last_sl_price = sl[-1]
        _, prev_sl_price = sl[-2]

        # CHoCH: structure flip — previously making lower-highs now breaks above, or vice versa
        making_lower_highs = last_sh_price < prev_sh_price
        making_higher_lows = last_sl_price > prev_sl_price

        if last_close > last_sh_price and prev_close <= last_sh_price:
            if making_lower_highs:
                return {'type': 'CHoCH_BULLISH', 'level': round(float(last_sh_price), 6), 'confirmed': True}
            return {'type': 'BOS_BULLISH', 'level': round(float(last_sh_price), 6), 'confirmed': True}

        if last_close < last_sl_price and prev_close >= last_sl_price:
            if making_higher_lows:
                return {'type': 'CHoCH_BEARISH', 'level': round(float(last_sl_price), 6), 'confirmed': True}
            return {'type': 'BOS_BEARISH', 'level': round(float(last_sl_price), 6), 'confirmed': True}

        # Unconfirmed — price approaching but not broken yet
        if last_close > last_sh_price * 0.998:
            return {'type': 'BOS_BULLISH', 'level': round(float(last_sh_price), 6), 'confirmed': False}
        if last_close < last_sl_price * 1.002:
            return {'type': 'BOS_BEARISH', 'level': round(float(last_sl_price), 6), 'confirmed': False}

        return {'type': 'NONE', 'level': 0, 'confirmed': False}

    # ─────────────────────────────────────────────────────────────────────
    # MARKET REGIME DETECTION  (Phase 2C)
    # ─────────────────────────────────────────────────────────────────────
    def detect_regime(self, df: pd.DataFrame) -> str:
        """
        Classifies current market regime using ATR volatility + EMA slope.
        Returns: 'TRENDING_BULL' | 'TRENDING_BEAR' | 'RANGING' | 'HIGH_VOLATILITY'
        """
        if len(df) < 30:
            return 'RANGING'

        last   = df.iloc[-1]
        prev10 = df.iloc[-10]

        atr_now  = float(last['atr']) if not pd.isna(last['atr']) else 0
        atr_mean = float(df['atr'].tail(20).mean()) if not df['atr'].tail(20).isna().all() else atr_now

        # High volatility: ATR is 1.5x its 20-period average
        if atr_now > atr_mean * 1.5:
            return 'HIGH_VOLATILITY'

        # EMA slope: compare current EMA20 to EMA20 10 candles ago
        ema20_now  = float(last['ema20'])  if not pd.isna(last['ema20'])  else 0
        ema20_prev = float(prev10['ema20']) if not pd.isna(prev10['ema20']) else ema20_now

        slope_pct = (ema20_now - ema20_prev) / ema20_prev * 100 if ema20_prev else 0

        if slope_pct > 0.5:
            return 'TRENDING_BULL'
        elif slope_pct < -0.5:
            return 'TRENDING_BEAR'
        return 'RANGING'

    # ─────────────────────────────────────────────────────────────────────
    # NEWS SENTIMENT  (free RSS — CoinDesk + CoinTelegraph + Decrypt)
    # No API key needed. Cached 10 min per coin.
    # ─────────────────────────────────────────────────────────────────────

    _RSS_FEEDS = [
        'https://www.coindesk.com/arc/outboundfeeds/rss/',
        'https://cointelegraph.com/rss',
        'https://decrypt.co/feed',
    ]

    # Keyword lists for headline sentiment scoring
    _BULL_WORDS = (
        'surge', 'rally', 'bullish', 'breakout', 'gain', 'rise', 'soar',
        'pump', 'buy', 'upgrade', 'adoption', 'partnership', 'launch',
        'approval', 'etf approved', 'record', 'high', 'institutional',
        'accumulate', 'support', 'recovery', 'bounce', 'moon',
    )
    _BEAR_WORDS = (
        'crash', 'drop', 'bearish', 'dump', 'sell', 'fear', 'hack',
        'exploit', 'ban', 'lawsuit', 'sec', 'regulation', 'fine',
        'fraud', 'scam', 'low', 'plunge', 'collapse', 'liquidat',
        'downgrade', 'delist', 'restrict', 'warning', 'risk',
    )

    # Shared RSS cache — fetched once, filtered per coin
    _rss_cache: dict = {'items': [], 'ts': 0}

    def _fetch_rss_headlines(self) -> list:
        """Fetches all RSS feeds and returns a flat list of title strings. Cached 10 min."""
        now = time.time()
        if now - self._rss_cache['ts'] < 600 and self._rss_cache['items']:
            return self._rss_cache['items']

        headlines = []
        for url in self._RSS_FEEDS:
            try:
                r = requests.get(url, timeout=6,
                                 headers={'User-Agent': 'Mozilla/5.0'})
                if not r.ok:
                    continue
                root = ET.fromstring(r.content)
                for item in root.iter('item'):
                    title = (item.findtext('title') or '').strip()
                    if title:
                        headlines.append(title.lower())
            except Exception:
                continue

        self._rss_cache['items'] = headlines
        self._rss_cache['ts'] = now
        return headlines

    def get_news_sentiment(self, symbol: str) -> str:
        """
        Scores recent crypto headlines from CoinDesk/CoinTelegraph/Decrypt
        for the given coin. Returns a one-line sentiment string for Claude.
        No API key required. Results cached 10 minutes per symbol.
        """
        now  = time.time()
        base = symbol.replace('/USDT', '').replace('/BTC', '').lower()
        cached = self._news_cache.get(base)
        if cached and now - cached['ts'] < 600:
            return cached['sentiment']

        try:
            all_headlines = self._fetch_rss_headlines()

            # Common coin name aliases so 'bitcoin' matches 'BTC' etc.
            aliases = {
                'btc': ['bitcoin', 'btc'],
                'eth': ['ethereum', 'eth', 'ether'],
                'sol': ['solana', 'sol'],
                'bnb': ['bnb', 'binance coin', 'binance'],
                'xrp': ['xrp', 'ripple'],
                'doge': ['dogecoin', 'doge'],
                'ada': ['cardano', 'ada'],
                'avax': ['avalanche', 'avax'],
                'link': ['chainlink', 'link'],
                'dot': ['polkadot', 'dot'],
                'matic': ['polygon', 'matic'],
                'arb': ['arbitrum', 'arb'],
                'op': ['optimism', ' op '],
                'sui': ['sui'],
                'inj': ['injective', 'inj'],
            }
            search_terms = aliases.get(base, [base])

            # Filter headlines mentioning this coin
            relevant = [h for h in all_headlines
                        if any(t in h for t in search_terms)]

            if not relevant:
                # Fallback: score general crypto sentiment from all headlines
                relevant = all_headlines[:20]

            bull = sum(1 for h in relevant for w in self._BULL_WORDS if w in h)
            bear = sum(1 for h in relevant for w in self._BEAR_WORDS if w in h)

            if bull == 0 and bear == 0:
                sentiment = 'No significant news'
            elif bear > bull * 1.5:
                label = 'NEGATIVE'
                preview = next((h for h in relevant
                                if any(w in h for w in self._BEAR_WORDS)), relevant[0])
                sentiment = f'{label} — {preview[:80]}'
            elif bull > bear * 1.5:
                label = 'POSITIVE'
                preview = next((h for h in relevant
                                if any(w in h for w in self._BULL_WORDS)), relevant[0])
                sentiment = f'{label} — {preview[:80]}'
            else:
                label = 'MIXED'
                sentiment = f'{label} — bull:{bull} bear:{bear} signals in {len(relevant)} headlines'

        except Exception as e:
            print(f'[News] {base}: {e}')
            sentiment = 'No recent news'

        self._news_cache[base] = {'sentiment': sentiment, 'ts': now}
        return sentiment

    # ─────────────────────────────────────────────────────────────────────
    # FUNDING RATE  (Phase 6 — Binance perpetual, free public endpoint)
    # ─────────────────────────────────────────────────────────────────────
    def get_funding_rate(self, symbol: str) -> dict:
        """
        Fetches current funding rate from Binance perp market (public, no key).
        Positive rate = longs pay shorts (market is over-leveraged long → bearish lean).
        Negative rate = shorts pay longs (market over-leveraged short → bullish lean).
        Cached 15 minutes per symbol.
        """
        now  = time.time()
        base = symbol.replace('/USDT', '').replace('/BTC', '')
        key  = f'fr_{base}'
        cached = self._news_cache.get(key)
        if cached and now - cached['ts'] < 900:
            return cached['data']

        result = {'rate': 0.0, 'label': 'Neutral', 'bias': 'NEUTRAL'}
        try:
            perp = base + 'USDT'
            r = requests.get(
                'https://fapi.binance.com/fapi/v1/premiumIndex',
                params={'symbol': perp},
                timeout=5,
            )
            if r.ok:
                data  = r.json()
                rate  = float(data.get('lastFundingRate', 0))
                pct   = round(rate * 100, 4)
                if rate > 0.0005:
                    label = f'HIGH POSITIVE ({pct}%) — longs over-leveraged, SHORT bias'
                    bias  = 'BEARISH'
                elif rate < -0.0005:
                    label = f'NEGATIVE ({pct}%) — shorts over-leveraged, LONG bias'
                    bias  = 'BULLISH'
                else:
                    label = f'Neutral ({pct}%)'
                    bias  = 'NEUTRAL'
                result = {'rate': pct, 'label': label, 'bias': bias}
        except Exception:
            pass

        self._news_cache[key] = {'data': result, 'ts': now}
        return result

    # ─────────────────────────────────────────────────────────────────────
    # FEAR & GREED
    # ─────────────────────────────────────────────────────────────────────
    def get_fear_and_greed(self) -> dict:
        now = time.time()
        if now - self._fng_cache['ts'] < 300:
            return self._fng_cache
        try:
            r    = with_retry(lambda: requests.get('https://api.alternative.me/fng/?limit=1', timeout=5))
            data = r.json()['data'][0]
            val  = int(data['value'])
            label = data['value_classification']
            self._fng_cache = {'value': val, 'label': label, 'ts': now}
        except Exception:
            pass
        return self._fng_cache

    # ─────────────────────────────────────────────────────────────────────
    # TIMEFRAME BIAS
    # ─────────────────────────────────────────────────────────────────────
    def get_timeframe_bias(self, symbol: str) -> dict:
        biases = {}
        for tf in ['15m', '1h', '4h']:
            try:
                limit = 350
                df = self.get_ohlcv(symbol, tf, limit=limit)
                df = self.add_indicators(df)
                df = df.dropna(subset=['ema20', 'ema50', 'rsi'])
                if df.empty:
                    biases[tf] = 'NEUTRAL'
                    continue
                last = df.iloc[-1]
                lookback = min(5, len(df) - 1)
                ema20_prev = float(df.iloc[-(lookback + 1)]['ema20'])
                slope_up   = float(last['ema20']) > ema20_prev
                if last['close'] > last['ema20'] > last['ema50'] and slope_up:
                    biases[tf] = 'BULLISH'
                elif last['close'] < last['ema20'] < last['ema50'] and not slope_up:
                    biases[tf] = 'BEARISH'
                else:
                    biases[tf] = 'NEUTRAL'
            except Exception:
                biases[tf] = 'NEUTRAL'

        bulls = sum(1 for b in biases.values() if b == 'BULLISH')
        bears = sum(1 for b in biases.values() if b == 'BEARISH')
        if bulls >= 2:   overall = 'BULLISH'
        elif bears >= 2: overall = 'BEARISH'
        else:            overall = 'MIXED'
        return {'per_tf': biases, 'overall': overall, 'agreement': max(bulls, bears)}

    # ─────────────────────────────────────────────────────────────────────
    # FULL ANALYSIS (main entry point)
    # ─────────────────────────────────────────────────────────────────────
    def get_full_analysis(self, symbol: str, timeframe: str = '1h') -> dict:
        df = self.get_ohlcv(symbol, timeframe, limit=350)
        df = self.add_indicators(df)
        df = df.dropna(subset=['ema20', 'ema50', 'rsi', 'atr', 'macd'])
        if df.empty:
            raise ValueError(f'Not enough candle data for {symbol}')

        last    = df.iloc[-1]
        prev24  = df.iloc[-24] if len(df) >= 24 else df.iloc[0]
        change24h = (last['close'] - prev24['close']) / prev24['close'] * 100

        fng            = self.get_fear_and_greed()
        tf_bias        = self.get_timeframe_bias(symbol)
        volume_signal  = self.get_volume_signal(df)
        volume_trend   = self.get_volume_trend(df)
        rsi_div        = self.detect_rsi_divergence(df)
        candle_pattern = self.detect_candle_pattern(df)
        news_sentiment = self.get_news_sentiment(symbol)
        bos_choch      = self.detect_bos_choch(df)
        regime         = self.detect_regime(df)
        funding_rate   = self.get_funding_rate(symbol)

        return {
            'symbol':         symbol,
            'timeframe':      timeframe,
            'timestamp':      str(df.index[-1]),
            'price':          float(last['close']),
            'change24h':      round(float(change24h), 2),
            'rsi':            round(float(last['rsi']), 1),
            'macd':           round(float(last['macd']), 6),
            'macd_sig':       round(float(last['macd_sig']), 6),
            'atr':            round(float(last['atr']), 6),
            'ema20':          round(float(last['ema20']), 6),
            'ema50':          round(float(last['ema50']), 6),
            'ema200':         None if pd.isna(last['ema200']) else round(float(last['ema200']), 6),
            'bb_upper':       round(float(last['bb_upper']), 6),
            'bb_lower':       round(float(last['bb_lower']), 6),
            'volume':         float(last['volume']),
            'volume_sma':     float(last['volume_sma']) if not pd.isna(last['volume_sma']) else 0,
            'volume_signal':  volume_signal,
            'volume_trend':   volume_trend,
            'rsi_divergence': rsi_div,
            'candle_pattern':  candle_pattern,
            'news_sentiment':  news_sentiment,
            'structure':       self.market_structure(df),
            'timeframe_bias': tf_bias,
            'fear_greed':     fng,
            'liquidity':      self.find_liquidity_levels(df),
            'order_blocks':   self.find_order_blocks(df),
            'fvgs':           self.find_fvgs(df),
            'bos_choch':      bos_choch,
            'regime':         regime,
            'funding_rate':   funding_rate,
            'candles':        df.tail(80)[['open', 'high', 'low', 'close', 'volume']].to_dict('records'),
        }