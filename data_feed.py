import ccxt
import pandas as pd
import pandas_ta_classic as ta
import requests
import time

class DataFeed:
    def __init__(self, exchange: ccxt.Exchange):
        self.exchange = exchange
        self._fng_cache = {'value': 50, 'label': 'Neutral', 'ts': 0}

    def get_ohlcv(self, symbol: str, timeframe: str = '1h', limit: int = 250) -> pd.DataFrame:
        raw = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        df = pd.DataFrame(raw, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        return df

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df['ema20'] = ta.ema(df['close'], length=20)
        df['ema50'] = ta.ema(df['close'], length=50)
        ema200 = ta.ema(df['close'], length=200)
        df['ema200'] = ema200 if ema200 is not None else df['ema50']
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        macd = ta.macd(df['close'])
        df['macd'] = macd['MACD_12_26_9']
        df['macd_sig'] = macd['MACDs_12_26_9']
        bb = ta.bbands(df['close'])
        bb_upper_col = [c for c in bb.columns if c.startswith('BBU')][0]
        bb_lower_col = [c for c in bb.columns if c.startswith('BBL')][0]
        df['bb_upper'] = bb[bb_upper_col]
        df['bb_lower'] = bb[bb_lower_col]
        df['volume_sma'] = ta.sma(df['volume'], length=20)
        return df

    def get_volume_signal(self, df: pd.DataFrame) -> str:
        last = df.iloc[-1]
        if pd.isna(last['volume_sma']) or last['volume_sma'] == 0:
            return 'NORMAL'
        ratio = last['volume'] / last['volume_sma']
        if ratio >= 1.5:
            return 'HIGH'
        elif ratio <= 0.7:
            return 'LOW'
        return 'NORMAL'

    def detect_rsi_divergence(self, df: pd.DataFrame) -> str:
        if len(df) < 5:
            return 'NONE'
        recent = df.tail(20)
        price_high_idx = recent['close'].idxmax()
        price_low_idx = recent['close'].idxmin()
        prev = recent.iloc[:-5]
        if prev.empty:
            return 'NONE'
        prev_high_idx = prev['close'].idxmax()
        prev_low_idx = prev['close'].idxmin()
        try:
            if (recent.loc[price_high_idx, 'close'] > recent.loc[prev_high_idx, 'close'] and
                    recent.loc[price_high_idx, 'rsi'] < recent.loc[prev_high_idx, 'rsi']):
                return 'BEARISH'
            if (recent.loc[price_low_idx, 'close'] < recent.loc[prev_low_idx, 'close'] and
                    recent.loc[price_low_idx, 'rsi'] > recent.loc[prev_low_idx, 'rsi']):
                return 'BULLISH'
        except Exception:
            pass
        return 'NONE'

    # SSL = Sell-Side Liquidity = equal lows (stops below lows)
    # BSL = Buy-Side Liquidity = equal highs (stops above highs)
    def find_liquidity_levels(self, df: pd.DataFrame) -> dict:
        recent = df.tail(30)
        bsl = recent['high'].nlargest(3).tolist()   # Buy-Side Liquidity: highs where shorts stops sit
        ssl = recent['low'].nsmallest(3).tolist()    # Sell-Side Liquidity: lows where longs stops sit
        return {'bsl': bsl, 'ssl': ssl}

    def find_order_blocks(self, df: pd.DataFrame) -> list:
        obs = []
        for i in range(5, len(df) - 3):
            candle = df.iloc[i]
            body_size = abs(candle['close'] - candle['open']) / candle['open']
            if body_size > 0.004:
                next3 = df.iloc[i + 1:i + 4]
                move = (next3['close'].iloc[-1] - candle['close']) / candle['close']
                if abs(move) > 0.008:
                    obs.append({
                        'price': float(candle['open']),
                        'type': 'bullish' if candle['close'] > candle['open'] else 'bearish',
                        'time': str(df.index[i])
                    })
        return obs[-3:]

    def find_fvgs(self, df: pd.DataFrame) -> list:
        fvgs = []
        for i in range(1, len(df) - 1):
            prev = df.iloc[i - 1]
            nxt = df.iloc[i + 1]
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

    def get_fear_and_greed(self) -> dict:
        now = time.time()
        if now - self._fng_cache['ts'] < 300:
            return self._fng_cache
        try:
            r = requests.get('https://api.alternative.me/fng/?limit=1', timeout=5)
            data = r.json()['data'][0]
            val = int(data['value'])
            label = data['value_classification']
            self._fng_cache = {'value': val, 'label': label, 'ts': now}
        except Exception:
            pass
        return self._fng_cache

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
        if bulls >= 2:
            overall = 'BULLISH'
        elif bears >= 2:
            overall = 'BEARISH'
        else:
            overall = 'MIXED'
        return {'per_tf': biases, 'overall': overall, 'agreement': max(bulls, bears)}

    def get_full_analysis(self, symbol: str, timeframe: str = '1h') -> dict:
        df = self.get_ohlcv(symbol, timeframe, limit=250)
        df = self.add_indicators(df)
        df = df.dropna(subset=['ema20', 'ema50', 'rsi', 'atr', 'macd'])
        if df.empty:
            raise ValueError(f'Not enough candle data for {symbol}')
        last = df.iloc[-1]
        prev24 = df.iloc[-24] if len(df) >= 24 else df.iloc[0]
        change24h = (last['close'] - prev24['close']) / prev24['close'] * 100
        fng = self.get_fear_and_greed()
        tf_bias = self.get_timeframe_bias(symbol)
        volume_signal = self.get_volume_signal(df)
        rsi_div = self.detect_rsi_divergence(df)

        return {
            'symbol': symbol,
            'timeframe': timeframe,
            'timestamp': str(df.index[-1]),
            'price': float(last['close']),
            'change24h': round(float(change24h), 2),
            'rsi': round(float(last['rsi']), 1),
            'macd': round(float(last['macd']), 6),
            'macd_sig': round(float(last['macd_sig']), 6),
            'atr': round(float(last['atr']), 6),
            'ema20': round(float(last['ema20']), 6),
            'ema50': round(float(last['ema50']), 6),
            'ema200': round(float(last['ema200']), 6),
            'bb_upper': round(float(last['bb_upper']), 6),
            'bb_lower': round(float(last['bb_lower']), 6),
            'volume': float(last['volume']),
            'volume_sma': float(last['volume_sma']) if not pd.isna(last['volume_sma']) else 0,
            'volume_signal': volume_signal,
            'rsi_divergence': rsi_div,
            'structure': self.market_structure(df),
            'timeframe_bias': tf_bias,
            'fear_greed': fng,
            'liquidity': self.find_liquidity_levels(df),
            'order_blocks': self.find_order_blocks(df),
            'fvgs': self.find_fvgs(df),
            'candles': df.tail(80)[['open', 'high', 'low', 'close', 'volume']].to_dict('records'),
        }
