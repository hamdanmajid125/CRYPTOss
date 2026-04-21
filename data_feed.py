import ccxt
import pandas as pd
import pandas_ta_classic as ta
import requests
import time

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
    def get_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 250) -> pd.DataFrame:
        raw = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['ema20']  = ta.ema(df['close'], length=20)
        df['ema50']  = ta.ema(df['close'], length=50)
        ema200       = ta.ema(df['close'], length=200)
        df['ema200'] = ema200 if ema200 is not None else df['ema50']
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
        if len(df) < 5:
            return 'NONE'
        recent = df.tail(20)
        price_high_idx = recent['close'].idxmax()
        price_low_idx  = recent['close'].idxmin()
        prev = recent.iloc[:-5]
        if prev.empty:
            return 'NONE'
        prev_high_idx = prev['close'].idxmax()
        prev_low_idx  = prev['close'].idxmin()
        try:
            if (recent.loc[price_high_idx, 'close'] > recent.loc[prev_high_idx, 'close'] and
                    recent.loc[price_high_idx, 'rsi']   < recent.loc[prev_high_idx, 'rsi']):
                return 'BEARISH DIVERGENCE — price up, RSI down, reversal likely'
            if (recent.loc[price_low_idx, 'close'] < recent.loc[prev_low_idx, 'close'] and
                    recent.loc[price_low_idx, 'rsi']  > recent.loc[prev_low_idx, 'rsi']):
                return 'BULLISH DIVERGENCE — price down, RSI up, reversal likely'
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
            candle    = df.iloc[i]
            body_size = abs(candle['close'] - candle['open']) / candle['open']
            if body_size > 0.004:
                next3 = df.iloc[i + 1:i + 4]
                move  = (next3['close'].iloc[-1] - candle['close']) / candle['close']
                if abs(move) > 0.008:
                    obs.append({
                        'price': float(candle['open']),
                        'type':  'bullish' if candle['close'] > candle['open'] else 'bearish',
                        'time':  str(df.index[i])
                    })
        return obs[-3:]

    def find_fvgs(self, df: pd.DataFrame) -> list:
        fvgs = []
        for i in range(1, len(df) - 1):
            prev = df.iloc[i - 1]
            nxt  = df.iloc[i + 1]
            if nxt['low'] > prev['high']:
                fvgs.append({'type': 'bull', 'top': float(nxt['low']), 'bot': float(prev['high'])})
            if nxt['high'] < prev['low']:
                fvgs.append({'type': 'bear', 'top': float(prev['low']), 'bot': float(nxt['high'])})
        return fvgs[-4:]

    def market_structure(self, df: pd.DataFrame) -> str:
        last = df.iloc[-1]
        if last['close'] > last['ema20'] > last['ema50']:
            return 'Bullish'
        elif last['close'] < last['ema20'] < last['ema50']:
            return 'Bearish'
        return 'Ranging'

    # ─────────────────────────────────────────────────────────────────────
    # NEWS SENTIMENT  (CryptoPanic — free public endpoint, no key needed)
    # ─────────────────────────────────────────────────────────────────────
    def get_news_sentiment(self, symbol: str) -> str:
        """
        Fetches recent news for a coin from CryptoPanic and returns a
        one-line sentiment string passed to the Claude prompt.
        Results cached 10 minutes per symbol to avoid hammering the API.
        """
        now  = time.time()
        base = symbol.replace('/USDT', '').replace('/BTC', '')
        cached = self._news_cache.get(base)
        if cached and now - cached['ts'] < 600:
            return cached['sentiment']

        try:
            params = {'public': 'true', 'currencies': base, 'kind': 'news', 'limit': 5}
            if self._cryptopanic_token:
                params['auth_token'] = self._cryptopanic_token

            r = requests.get(
                'https://cryptopanic.com/api/v1/posts/',
                params=params,
                timeout=6,
            )
            if not r.ok:
                raise ValueError(f'HTTP {r.status_code}')

            results = r.json().get('results', [])
            if not results:
                sentiment = 'No recent news'
            else:
                pos = sum(a.get('votes', {}).get('positive', 0) for a in results)
                neg = sum(a.get('votes', {}).get('negative', 0) for a in results)
                headlines = ' | '.join(a['title'][:60] for a in results[:2])

                if neg > pos * 1.5:
                    label = 'NEGATIVE'
                elif pos > neg * 1.5:
                    label = 'POSITIVE'
                else:
                    label = 'MIXED'

                sentiment = f'{label} — {headlines}'

        except Exception as e:
            print(f'[News] {base}: {e}')
            sentiment = 'No recent news'

        self._news_cache[base] = {'sentiment': sentiment, 'ts': now}
        return sentiment

    # ─────────────────────────────────────────────────────────────────────
    # FEAR & GREED
    # ─────────────────────────────────────────────────────────────────────
    def get_fear_and_greed(self) -> dict:
        now = time.time()
        if now - self._fng_cache['ts'] < 300:
            return self._fng_cache
        try:
            r    = requests.get('https://api.alternative.me/fng/?limit=1', timeout=5)
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
                limit = 100 if tf == '4h' else 150
                df = self.get_ohlcv(symbol, tf, limit=limit)
                df = self.add_indicators(df)
                df = df.dropna(subset=['ema20', 'ema50', 'rsi'])
                if df.empty:
                    biases[tf] = 'NEUTRAL'
                    continue
                last = df.iloc[-1]
                if last['close'] > last['ema20'] > last['ema50'] and last['rsi'] > 50:
                    biases[tf] = 'BULLISH'
                elif last['close'] < last['ema20'] < last['ema50'] and last['rsi'] < 50:
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
        df = self.get_ohlcv(symbol, timeframe, limit=250)
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
            'ema200':         round(float(last['ema200']), 6),
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
            'candles':        df.tail(80)[['open', 'high', 'low', 'close', 'volume']].to_dict('records'),
        }