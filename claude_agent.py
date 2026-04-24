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
            return 'London (high volatility, prefer trend trades)'
        elif 13 <= hour < 21:
            return 'New York (highest volume, best for breakouts)'
        elif 0 <= hour < 8:
            return 'Asia (lower volatility, range/grid strategies better)'
        return 'London/NY Overlap (BEST session — highest confluence probability)'

    def _score_confidence(self, data: dict, tf_bias: dict, fng: dict) -> int:
        score = 50

        # Timeframe agreement (most important)
        agreement = tf_bias.get('agreement', 0)
        if agreement == 3:   score += 20
        elif agreement == 2: score += 8
        else:                score -= 20  # No trade if TFs conflict

        # Volume confirmation
        vol = data.get('volume_signal', 'NORMAL')
        if 'VERY HIGH'  in vol: score += 15
        elif 'HIGH'     in vol: score += 8
        elif 'VERY LOW' in vol: score -= 15
        elif 'LOW'      in vol: score -= 8

        # Volume trend
        vt = data.get('volume_trend', 'FLAT')
        if 'INCREASING' in vt: score += 5
        elif 'DECREASING' in vt: score -= 5

        # Fear & Greed contrarian boost
        fng_val = fng.get('value', 50)
        overall = tf_bias.get('overall', 'MIXED')
        if fng_val < 20 and overall == 'BULLISH': score += 12   # Extreme fear + bullish = great buy
        elif fng_val > 80 and overall == 'BEARISH': score += 12  # Extreme greed + bearish = great short
        elif fng_val < 35 and overall == 'BULLISH': score += 6
        elif fng_val > 65 and overall == 'BEARISH': score += 6

        # RSI positioning (avoid chasing overbought/oversold)
        rsi = data.get('rsi', 50)
        if overall == 'BULLISH' and 40 <= rsi <= 65: score += 10   # Sweet spot for longs
        elif overall == 'BEARISH' and 35 <= rsi <= 60: score += 10  # Sweet spot for shorts
        elif rsi > 78 and overall == 'BULLISH':  score -= 15  # Too overbought for longs
        elif rsi < 22 and overall == 'BEARISH':  score -= 15  # Too oversold for shorts

        # RSI Divergence (powerful reversal signal)
        rsi_div = data.get('rsi_divergence', 'NONE')
        if 'BULLISH' in rsi_div and overall == 'BULLISH': score += 12
        elif 'BEARISH' in rsi_div and overall == 'BEARISH': score += 12

        # Candlestick pattern boost
        pattern = data.get('candle_pattern', '')
        high_conf_patterns = ['Bullish Engulfing', 'Bearish Engulfing', 'Morning Star', 'Evening Star',
                              'Bullish Marubozu', 'Bearish Marubozu']
        med_conf_patterns  = ['Hammer', 'Shooting Star']
        if any(p in pattern for p in high_conf_patterns): score += 12
        elif any(p in pattern for p in med_conf_patterns): score += 6
        elif 'Doji' in pattern:  score -= 5   # Indecision — wait

        # SMC: FVG presence
        fvgs = data.get('fvgs', [])
        price = data.get('price', 0)
        in_fvg = any(f['bot'] <= price <= f['top'] for f in fvgs)
        if in_fvg: score += 8

        # SMC: near order block
        obs = data.get('order_blocks', [])
        near_ob = any(abs(o['price'] - price) / price < 0.005 for o in obs)
        if near_ob: score += 8

        # EMA200 alignment
        ema200 = data.get('ema200', price)
        if overall == 'BULLISH' and price > ema200: score += 5   # Macro aligned
        elif overall == 'BULLISH' and price < ema200: score -= 10  # Fighting macro trend
        elif overall == 'BEARISH' and price < ema200: score += 5
        elif overall == 'BEARISH' and price > ema200: score -= 10

        # MACD alignment
        macd     = data.get('macd', 0)
        macd_sig = data.get('macd_sig', 0)
        if overall == 'BULLISH' and macd > macd_sig: score += 5
        elif overall == 'BEARISH' and macd < macd_sig: score += 5

        return max(0, min(100, score))

    # ─────────────────────────────────────────────────────────────────────────
    # THE BEST TRADING PROMPT  (completely rewritten)
    # ─────────────────────────────────────────────────────────────────────────
    def build_prompt(self, data: dict) -> str:
        liq       = data['liquidity']
        obs       = data['order_blocks']
        fvgs      = data['fvgs']
        tf_bias   = data['timeframe_bias']
        fng       = data['fear_greed']
        price     = data['price']
        atr       = data['atr']
        per_tf    = tf_bias.get('per_tf', {})
        bos_choch    = data.get('bos_choch', {'type': 'NONE', 'level': 0, 'confirmed': False})
        regime       = data.get('regime', 'RANGING')
        funding_rate = data.get('funding_rate', {'rate': 0.0, 'label': 'Neutral', 'bias': 'NEUTRAL'})

        pre_score      = self._score_confidence(data, tf_bias, fng)
        candle_pattern = data.get('candle_pattern', 'No pattern detected')
        volume_signal  = data.get('volume_signal', 'NORMAL')
        volume_trend   = data.get('volume_trend', 'FLAT')
        rsi_div        = data.get('rsi_divergence', 'NONE')

        tf_agreement = tf_bias.get('agreement', 0)

        # Dynamic guidance based on Fear & Greed
        fng_val = fng['value']
        if fng_val < 20:
            fng_guidance = ('🔴 EXTREME FEAR (< 20) — Markets are panicking. This is historically the BEST '
                            'time to buy. Only take HIGH CONFIDENCE longs at key SMC demand zones. '
                            'Avoid shorts — risk/reward is poor.')
        elif fng_val < 35:
            fng_guidance = ('🟠 FEAR (< 35) — Cautious sentiment. Favor long setups at SMC support. '
                            'Avoid longs near resistance. Be patient.')
        elif fng_val > 80:
            fng_guidance = ('🔴 EXTREME GREED (> 80) — Markets are euphoric. High risk of reversal. '
                            'Only take SHORT setups at strong SMC supply zones. '
                            'Avoid longs — you are buying near tops.')
        elif fng_val > 65:
            fng_guidance = ('🟡 GREED (> 65) — Elevated optimism. Tighten SL on longs. '
                            'Short setups at resistance are valid.')
        else:
            fng_guidance = '🟢 NEUTRAL (35-65) — Both directions valid. Let SMC + price action decide.'

        # Strict TF agreement warning
        if tf_agreement < 2:
            tf_warning = ('⛔ CRITICAL: Only {}/3 timeframes agree. This is a HIGH-RISK zone. '
                          'Output WAIT unless you see an exceptionally clear SMC setup with '
                          'ALL other factors aligned.').format(tf_agreement)
        else:
            tf_warning = f'✅ TIMEFRAME CONFLUENCE: {tf_agreement}/3 timeframes aligned — proceed with analysis.'

        # Candle pattern guidance
        if 'Engulfing' in candle_pattern or 'Star' in candle_pattern or 'Marubozu' in candle_pattern:
            pattern_boost = f'⭐ HIGH-CONFIDENCE PATTERN: {candle_pattern} — Add +10 to confidence if aligned with bias.'
        elif 'Doji' in candle_pattern:
            pattern_boost = f'⚠️ DOJI: Market is indecisive. Reduce confidence by 10. Wait for next candle confirmation.'
        else:
            pattern_boost = f'Pattern: {candle_pattern}'

        return f'''You are a professional institutional crypto trader with 15 years of experience.
You specialize in Smart Money Concepts (SMC), ICT methodology, multi-timeframe confluence,
and candlestick analysis. Your #1 rule: PROTECT CAPITAL FIRST, profit second.
You only take trades when multiple factors align perfectly. When in doubt, you WAIT.

══════════════════════════════════════════════════════
ASSET: {data['symbol']} | TIMEFRAME: {data['timeframe']} | SESSION: {self._trading_session()}
PRE-COMPUTED CONFIDENCE SCORE: {pre_score}/100
══════════════════════════════════════════════════════

━━━ STEP 1: MULTI-TIMEFRAME STRUCTURE ━━━
15m Bias : {per_tf.get('15m', 'NEUTRAL')}
1h  Bias : {per_tf.get('1h',  'NEUTRAL')}
4h  Bias : {per_tf.get('4h',  'NEUTRAL')}
Overall  : {tf_bias['overall']} ({tf_bias['agreement']}/3 timeframes agree)
{tf_warning}

RULE: If overall = MIXED → output WAIT. No exceptions.
RULE: LONG only if overall = BULLISH. SHORT only if overall = BEARISH.

━━━ STEP 2: TREND & EMA STRUCTURE ━━━
Price  : {price}
EMA20  : {data['ema20']}  → Price {'ABOVE ✅' if price > data['ema20'] else 'BELOW ❌'}
EMA50  : {data['ema50']}  → Price {'ABOVE ✅' if price > data['ema50'] else 'BELOW ❌'}
EMA200 : {data['ema200']} → Price {'ABOVE ✅ (macro bullish)' if price > data['ema200'] else 'BELOW ❌ (macro bearish)'}
24h Change: {data['change24h']}%

EMA200 RULE: Trading AGAINST EMA200 = -10 confidence. Trading WITH EMA200 = +5 confidence.

━━━ STEP 3: MOMENTUM INDICATORS ━━━
RSI(14)        : {data['rsi']} {'⚠️ OVERBOUGHT — avoid longs, shorts preferred' if data['rsi'] > 72 else '✅ OVERSOLD — longs preferred, avoid shorts' if data['rsi'] < 28 else '✅ Neutral zone'}
RSI Divergence : {rsi_div}
  → Bullish divergence + bullish TF bias = VERY HIGH confidence long
  → Bearish divergence + bearish TF bias = VERY HIGH confidence short

MACD           : {data['macd']} vs Signal {data['macd_sig']}
  → {'✅ BULLISH crossover — momentum supports longs' if data['macd'] > data['macd_sig'] else '❌ BEARISH crossover — momentum supports shorts'}

Bollinger Bands: Upper {data['bb_upper']} | Lower {data['bb_lower']}
  → {'⚠️ Price near UPPER band — resistance zone, avoid longs' if price > data['bb_upper'] * 0.998 else '✅ Price near LOWER band — support zone, favor longs' if price < data['bb_lower'] * 1.002 else 'Price in mid-band range'}

━━━ STEP 4: VOLUME ANALYSIS ━━━
Volume Signal : {volume_signal}
Volume Trend  : {volume_trend}

VOLUME RULES (NON-NEGOTIABLE):
→ VERY LOW volume = DO NOT TRADE regardless of setup quality
→ HIGH/VERY HIGH volume = confirms the move, add +8 confidence
→ INCREASING trend + breakout = strong momentum confirmation
→ DECREASING trend near resistance = reversal likely

━━━ STEP 5: CANDLESTICK PATTERNS ━━━
{pattern_boost}

CANDLESTICK RULES:
→ Bullish/Bearish Engulfing at SMC level = VERY HIGH confidence signal
→ Morning/Evening Star = Strong 3-candle reversal, high confidence
→ Hammer at demand zone = HIGH confidence long
→ Shooting Star at supply zone = HIGH confidence short
→ Doji = WAIT for next candle before deciding
→ No pattern = rely entirely on SMC + TF confluence

━━━ STEP 6: SMART MONEY CONCEPTS (SMC) ━━━
Market Regime  : {regime}
  → TRENDING_BULL/BEAR = ride momentum  |  RANGING = wait for OB/FVG touch  |  HIGH_VOLATILITY = reduce size

BOS / CHoCH    : {bos_choch['type']} @ {bos_choch['level']} ({'CONFIRMED ✅' if bos_choch['confirmed'] else 'UNCONFIRMED ⚠️'})
  → CHoCH = structure flip, strongest reversal signal — HIGH confidence if confirmed
  → BOS   = continuation of existing trend — trade in direction of break
  → NONE  = no structure break detected, avoid breakout entries

Buy-Side Liquidity  (BSL — stops above highs)  : {', '.join(str(round(p, 4)) for p in liq['bsl'])}
Sell-Side Liquidity (SSL — stops below lows)   : {', '.join(str(round(p, 4)) for p in liq['ssl'])}

Order Blocks detected ({len(obs)}) : {', '.join(set(o['type'] for o in obs)) if obs else '❌ NONE — reduce confidence -15'}
  → Bullish OB: strong demand zone, price should bounce here for longs
  → Bearish OB: strong supply zone, price should reject here for shorts
  → No OBs = weaker setup, add -15 to confidence

Fair Value Gaps (FVGs) ({len(fvgs)}) : {', '.join(set(f['type'] for f in fvgs)) if fvgs else '❌ NONE'}
  → Price inside a FVG = likely to fill it (strong magnet)
  → Bull FVG = price tends to rally UP through it
  → Bear FVG = price tends to fall DOWN through it

BEST SMC SETUPS (in order of confidence):
1. Liquidity Sweep + Break of Structure (BOS) → HIGHEST confidence
2. Price returns to OB after BOS → HIGH confidence
3. Price fills FVG + candle pattern → HIGH confidence
4. OB + FVG overlap = "Golden Zone" → VERY HIGH confidence

━━━ STEP 7: ATR-BASED SL/TP CALCULATION ━━━
ATR(14): {atr}

STOP LOSS RULES:
→ Place SL BEYOND the nearest Order Block or key swing level
→ Never place SL at exactly round numbers (market will hunt them)
→ Minimum SL distance: {round(atr * 1.2, 6)} | Maximum: {round(atr * 2.5, 6)}
→ SL too tight = gets stopped out by noise → lose money
→ SL too wide = bad risk/reward → also lose money

TAKE PROFIT RULES (minimum 1:2 Risk/Reward — NEVER trade 1:1):
→ TP1 : {round(atr * 2.2, 6)} from entry (take 40% of position here, move SL to breakeven)
→ TP2 : {round(atr * 4.0, 6)} from entry (take 40% here)
→ TP3 : {round(atr * 7.0, 6)} from entry (let 20% run — the "runner")

━━━ STEP 8: MARKET SENTIMENT ━━━
Fear & Greed Index: {fng_val}/100 — {fng['label']}
{fng_guidance}

Funding Rate: {funding_rate['label']}
  → Positive funding = market over-long → fade longs, prefer shorts
  → Negative funding = market over-short → fade shorts, prefer longs
  → Extreme funding (>0.1%) = HIGH contrarian signal, add ±8 confidence

━━━ STEP 9: SESSION ADVANTAGE ━━━
Session: {self._trading_session()}
→ London/NY Overlap: BEST for breakouts and trending moves
→ NY Session: High volume, great for momentum trades
→ London Open: Good for liquidity sweeps and reversals
→ Asia: Low volume, choppy — prefer smaller position size or skip

━━━ FINAL DECISION RULES (READ CAREFULLY) ━━━
✅ OUTPUT LONG ONLY IF ALL TRUE:
   - TF overall = BULLISH (≥ 2/3 agree)
   - RSI < 70 (not overbought)
   - MACD bullish or turning bullish
   - Volume HIGH or NORMAL (not VERY LOW)
   - At least one: OB, FVG, or candle pattern confirming
   - Final confidence ≥ 72

✅ OUTPUT SHORT ONLY IF ALL TRUE:
   - TF overall = BEARISH (≥ 2/3 agree)
   - RSI > 30 (not oversold)
   - MACD bearish or turning bearish
   - Volume HIGH or NORMAL (not VERY LOW)
   - At least one: OB, FVG, or candle pattern confirming
   - Final confidence ≥ 72

⛔ OUTPUT WAIT IF ANY:
   - TF overall = MIXED
   - Volume = VERY LOW
   - RSI > 78 and signal is LONG
   - RSI < 22 and signal is SHORT
   - Doji candle with no other confluence
   - Confidence < 72
   - You are unsure — WHEN IN DOUBT, WAIT. A skipped trade is NOT a lost trade.

Remember: The best traders miss many trades. They only take the perfect setup.
Every WAIT decision you make is protecting capital.

Respond ONLY with this exact JSON (no markdown, no preamble, no explanation):
{{"action":"LONG or SHORT or WAIT","confidence":0-100,"entry":{price},"sl":0.0,"tp1":0.0,"tp2":0.0,"tp3":0.0,"rr":"1:X.X","setup_type":"exact SMC setup name e.g. Bullish OB+FVG+Engulfing / Liquidity Sweep+BOS+Retest / WAIT-mixed-TF","session":"{self._trading_session()}","candle_pattern":"{candle_pattern}","timeframe_bias":"{tf_bias['overall']}","sentiment":"{fng['label']}","volume_signal":"{volume_signal}","reason":"3 sentences: (1) SMC structure and why this entry (2) Volume + candlestick + RSI confluence (3) Key risk factor and what would invalidate this trade","key_level":"most critical price level to watch","invalidation":"exact price that completely invalidates this setup — where you would cut the loss"}}'''

    # ─────────────────────────────────────────────────────────────────────────
    # GET SIGNAL
    # ─────────────────────────────────────────────────────────────────────────
    def get_signal(self, market_data: dict, deep: bool = False) -> Optional[Dict]:
        try:
            tf_bias   = market_data.get('timeframe_bias', {'overall': 'MIXED', 'agreement': 0})
            fng       = market_data.get('fear_greed', {'value': 50, 'label': 'Neutral'})
            pre_score = self._score_confidence(market_data, tf_bias, fng)

            # Use deep model only for very high pre-scores (save API cost)
            model = self.model_deep if (deep or pre_score >= 80) else self.model_fast

            response = self.client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{'role': 'user', 'content': self.build_prompt(market_data)}]
            )

            text = next(b.text for b in response.content if b.type == 'text').strip()
            text = text.replace('```json', '').replace('```', '').strip()
            signal = json.loads(text)

            signal['symbol']    = market_data['symbol']
            signal['price']     = market_data['price']
            signal['timestamp'] = market_data.get('timestamp', '')
            signal['pre_score'] = pre_score

            # ✅ Confidence gate raised to 72 (was 65 — too low!)
            if signal.get('action') in ('LONG', 'SHORT') and signal.get('confidence', 0) < 72:
                signal['action'] = 'WAIT'
                signal['reason'] = (f"Confidence {signal['confidence']}% below minimum 72. "
                                    + signal.get('reason', ''))

            return signal

        except Exception as e:
            print(f'[Claude] Error getting signal: {e}')
            return None

    def weekly_report(self, market_data_list: list) -> str:
        summary = '\n'.join([
            f"{d['symbol']}: ${d['price']} | RSI {d['rsi']} | {d['structure']} | F&G {d['fear_greed']['value']} | {d.get('candle_pattern','')}"
            for d in market_data_list
        ])
        response = self.client.messages.create(
            model=self.model_deep,
            max_tokens=2000,
            messages=[{'role': 'user', 'content': f'''You are a professional crypto hedge fund analyst.
Write a weekly market report covering these assets:
{summary}
Include: macro bias, best setups this week, key levels, risk factors, sentiment analysis.
Format as a professional trading report with clear sections.'''}]
        )
        return response.content[0].text