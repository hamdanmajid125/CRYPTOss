import json
import anthropic
from typing import Optional, Dict
import datetime


class ClaudeAgent:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_fast = 'claude-sonnet-4-6'
        self.model_deep = 'claude-opus-4-7'

    def _trading_session(self) -> str:
        hour = datetime.datetime.now(datetime.timezone.utc).hour
        if 8 <= hour < 12:
            return 'London'
        elif 13 <= hour < 21:
            return 'NY'
        elif 0 <= hour < 8:
            return 'Asia'
        return 'London/NY Overlap'

    def _score_confidence(self, data: dict, tf_bias: dict, fng: dict) -> int:
        score = 50
        agreement = tf_bias.get('agreement', 0)
        if agreement == 3:
            score += 15
        elif agreement == 2:
            score += 5
        else:
            score -= 15

        vol = data.get('volume_signal', 'NORMAL')
        if vol == 'HIGH':
            score += 10
        elif vol == 'LOW':
            score -= 8

        fng_val = fng.get('value', 50)
        overall = tf_bias.get('overall', 'MIXED')
        if fng_val < 25 and overall == 'BULLISH':
            score += 10
        elif fng_val > 75 and overall == 'BEARISH':
            score += 10

        rsi = data.get('rsi', 50)
        if overall == 'BULLISH' and rsi < 70:
            score += 8
        elif overall == 'BEARISH' and rsi > 30:
            score += 8

        fvgs = data.get('fvgs', [])
        price = data.get('price', 0)
        in_fvg = any(f['bot'] <= price <= f['top'] for f in fvgs)
        if in_fvg:
            score += 7

        obs = data.get('order_blocks', [])
        near_ob = any(abs(o['price'] - price) / price < 0.005 for o in obs)
        if near_ob:
            score += 5

        ema200 = data.get('ema200', price)
        if overall == 'BULLISH' and price < ema200:
            score -= 10
        elif overall == 'BEARISH' and price > ema200:
            score -= 10

        return max(0, min(100, score))

    def build_prompt(self, data: dict) -> str:
        liq = data['liquidity']
        obs = data['order_blocks']
        fvgs = data['fvgs']
        tf_bias = data['timeframe_bias']
        fng = data['fear_greed']
        price = data['price']
        atr = data['atr']
        per_tf = tf_bias.get('per_tf', {})
        pre_score = self._score_confidence(data, tf_bias, fng)

        fng_guidance = ''
        if fng['value'] < 25:
            fng_guidance = 'Extreme Fear — contrarian LONG opportunities preferred.'
        elif fng['value'] > 75:
            fng_guidance = 'Extreme Greed — be cautious, tighten SL on longs.'

        return f'''You are an elite institutional crypto trader using Smart Money Concepts (SMC), ICT methodology, and multi-timeframe confluence.

Analyze {data['symbol']} and generate a PRECISE trading signal with all required fields.

=== MULTI-TIMEFRAME BIAS ===
15m: {per_tf.get('15m', 'NEUTRAL')} | 1h: {per_tf.get('1h', 'NEUTRAL')} | 4h: {per_tf.get('4h', 'NEUTRAL')}
Overall Bias: {tf_bias['overall']} (agreement: {tf_bias['agreement']}/3 timeframes)
Rule: Only LONG/SHORT if ≥2 timeframes agree. Otherwise WAIT.

=== MARKET SENTIMENT ===
Fear & Greed Index: {fng['value']}/100 — {fng['label']}
{fng_guidance}

=== VOLUME ANALYSIS ===
Volume Signal: {data['volume_signal']} (current vs 20-SMA)
Note: HIGH volume confirms direction. LOW volume weakens signal.

=== PRICE DATA ({data['timeframe']}) ===
Current price: {price}
24h change: {data['change24h']}%
RSI(14): {data['rsi']} {'— OVERBOUGHT CAUTION' if data['rsi'] > 70 else '— OVERSOLD OPPORTUNITY' if data['rsi'] < 30 else ''}
RSI Divergence: {data.get('rsi_divergence', 'NONE')}
MACD: {data['macd']} | Signal: {data['macd_sig']} | {'BULLISH' if data['macd'] > data['macd_sig'] else 'BEARISH'} crossover
ATR(14): {atr} (use for SL/TP sizing)

=== EMA STRUCTURE ===
EMA20: {data['ema20']} — price {'ABOVE ✓' if price > data['ema20'] else 'BELOW ✗'}
EMA50: {data['ema50']} — price {'ABOVE ✓' if price > data['ema50'] else 'BELOW ✗'}
EMA200: {data['ema200']} — price {'ABOVE ✓ (macro bullish)' if price > data['ema200'] else 'BELOW ✗ (macro bearish)'}

=== BOLLINGER BANDS ===
Upper: {data['bb_upper']} | Lower: {data['bb_lower']}
Position: {'Near UPPER band — potential reversal zone' if price > data['bb_upper'] * 0.998 else 'Near LOWER band — potential support' if price < data['bb_lower'] * 1.002 else 'Mid-band — trending'}

=== SMART MONEY CONCEPTS ===
Buy-Side Liquidity (BSL — stops above highs): {', '.join(str(round(p, 2)) for p in liq['bsl'])}
Sell-Side Liquidity (SSL — stops below lows): {', '.join(str(round(p, 2)) for p in liq['ssl'])}
Order Blocks: {len(obs)} detected — {', '.join(set(o['type'] for o in obs)) if obs else 'none'}
Fair Value Gaps: {len(fvgs)} — {', '.join(set(f['type'] for f in fvgs)) if fvgs else 'none'}

=== SL/TP GUIDELINES (ATR-based) ===
Suggested SL distance: {round(atr * 1.5, 2)} (1.5x ATR)
TP1 target: {round(atr * 2.0, 2)} from entry (1:1.3 R:R minimum)
TP2 target: {round(atr * 3.5, 2)} from entry
TP3 target: {round(atr * 5.5, 2)} from entry (runner)

=== PRE-SCORED CONFIDENCE: {pre_score}/100 ===
Adjust final confidence based on your SMC analysis. Min 65 to output LONG/SHORT.

=== SESSION ===
Current: {self._trading_session()}

Respond ONLY with this exact JSON (no markdown, no preamble):
{{"action":"LONG or SHORT or WAIT","confidence":0-100,"entry":{price},"sl":number,"tp1":number,"tp2":number,"tp3":number,"rr":"1:X.X","timeframe_bias":"{tf_bias['overall']}","sentiment":"{fng['label']}","volume_signal":"{data['volume_signal']}","setup_type":"SMC setup name e.g. Bullish OB+FVG / Liquidity Sweep+MSB / BOS+Retest","session":"{self._trading_session()}","reason":"3 sentences: (1) SMC setup rationale (2) sentiment+volume confluence (3) key risk","key_level":"most important price to watch","invalidation":"price that invalidates this setup"}}'''

    def get_signal(self, market_data: dict, deep: bool = False) -> Optional[Dict]:
        try:
            tf_bias = market_data.get('timeframe_bias', {'overall': 'MIXED', 'agreement': 0})
            fng = market_data.get('fear_greed', {'value': 50, 'label': 'Neutral'})
            pre_score = self._score_confidence(market_data, tf_bias, fng)

            model = self.model_deep if deep else self.model_fast
            response = self.client.messages.create(
                model=model,
                max_tokens=700,
                messages=[{
                    'role': 'user',
                    'content': self.build_prompt(market_data)
                }]
            )
            text = response.content[0].text.strip()
            text = text.replace('```json', '').replace('```', '').strip()
            signal = json.loads(text)
            signal['symbol'] = market_data['symbol']
            signal['price'] = market_data['price']
            signal['timestamp'] = market_data.get('timestamp', '')
            signal['pre_score'] = pre_score

            # Enforce minimum confidence gate
            if signal.get('action') in ('LONG', 'SHORT') and signal.get('confidence', 0) < 65:
                signal['action'] = 'WAIT'
                signal['reason'] = f"Confidence {signal['confidence']}% below minimum 65. " + signal.get('reason', '')

            return signal
        except Exception as e:
            print(f'[Claude] Error: {e}')
            return None

    def weekly_report(self, symbols: list, market_data_list: list) -> str:
        summary = '\n'.join([
            f"{d['symbol']}: {d['price']} | RSI {d['rsi']} | {d['structure']} | F&G {d['fear_greed']['value']}"
            for d in market_data_list
        ])
        response = self.client.messages.create(
            model=self.model_deep,
            max_tokens=2000,
            messages=[{
                'role': 'user',
                'content': f'''You are a professional crypto hedge fund analyst.

Write a weekly market report covering these assets:
{summary}

Include: macro bias, key levels to watch, best setups this week, risk factors, sentiment analysis.
Format as a professional trading report with clear sections.'''
            }]
        )
        return response.content[0].text
