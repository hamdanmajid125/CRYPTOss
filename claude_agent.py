import json
import anthropic
from typing import Optional, Dict
import datetime


class ClaudeAgent:

    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model_fast = 'claude-sonnet-4-6'
        self.model_deep = 'claude-opus-4-5-20251101'

    def _trading_session(self) -> str:
        hour = datetime.datetime.now(datetime.timezone.utc).hour
        if 8 <= hour < 12:
            return 'London'
        elif 12 <= hour < 17:
            return 'London/NY Overlap'
        elif 17 <= hour < 21:
            return 'NY'
        else:
            return 'Asia'

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

        # RSI positioning — symmetric sweet spot 38-62 for both directions
        rsi = data.get('rsi', 50)
        if overall == 'BULLISH' and 38 <= rsi <= 62: score += 10
        elif overall == 'BEARISH' and 38 <= rsi <= 62: score += 10
        elif rsi > 75 and overall == 'BULLISH':  score -= 15  # Overbought, bad long entry
        elif rsi < 25 and overall == 'BEARISH':  score -= 15  # Oversold, bad short entry

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

        # SMC: FVG — direction-aware (being inside opposing FVG hurts, not helps)
        fvgs  = data.get('fvgs', [])
        price = data.get('price', 0)
        in_bull_fvg = any(f['bot'] <= price <= f['top'] for f in fvgs if f.get('type') == 'bullish')
        in_bear_fvg = any(f['bot'] <= price <= f['top'] for f in fvgs if f.get('type') == 'bearish')
        if overall == 'BULLISH':
            if in_bull_fvg: score += 10   # Price in demand FVG — good long entry
            if in_bear_fvg: score -= 12   # Price in supply FVG — bad long entry
        elif overall == 'BEARISH':
            if in_bear_fvg: score += 10   # Price in supply FVG — good short entry
            if in_bull_fvg: score -= 12   # Price in demand FVG — bad short entry

        # SMC: order block — direction-aware zone check
        obs = data.get('order_blocks', [])
        for o in obs:
            top = o.get('top', o['price'] * 1.001)
            bot = o.get('bot', o['price'] * 0.999)
            in_zone = bot <= price <= top
            if not in_zone:
                continue
            if overall == 'BULLISH' and o['type'] == 'bullish':
                score += 12   # Inside demand OB — ideal long entry
            elif overall == 'BEARISH' and o['type'] == 'bearish':
                score += 12   # Inside supply OB — ideal short entry
            elif overall == 'BULLISH' and o['type'] == 'bearish':
                score -= 12   # Inside supply OB — entering into resistance
            elif overall == 'BEARISH' and o['type'] == 'bullish':
                score -= 12   # Inside demand OB — entering into support

        # 4H veto: strong timeframe contradiction overrides weaker agreement
        per_tf = tf_bias.get('per_tf', {})
        if per_tf.get('4h') == 'BEARISH' and overall == 'BULLISH':
            score -= 25   # 4H bearish kills long bias
        elif per_tf.get('4h') == 'BULLISH' and overall == 'BEARISH':
            score -= 25   # 4H bullish kills short bias

        # EMA200 alignment (skip if not enough candle data)
        ema200 = data.get('ema200')
        if ema200 is not None:
            if overall == 'BULLISH' and price > ema200:   score += 5
            elif overall == 'BULLISH' and price < ema200: score -= 10
            elif overall == 'BEARISH' and price < ema200: score += 5
            elif overall == 'BEARISH' and price > ema200: score -= 10

        # MACD alignment — reward aligned, penalise opposing
        macd     = data.get('macd', 0)
        macd_sig = data.get('macd_sig', 0)
        if overall == 'BULLISH' and macd > macd_sig:   score += 5
        elif overall == 'BEARISH' and macd < macd_sig: score += 5
        elif overall == 'BULLISH' and macd < macd_sig: score -= 8
        elif overall == 'BEARISH' and macd > macd_sig: score -= 8

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
        ema200         = data.get('ema200')
        ema200_str     = 'N/A (not enough data — 200+ candles required)' if ema200 is None else str(ema200)
        ema200_arrow   = ('⚠️ Unavailable' if ema200 is None else
                          'ABOVE ✅ (macro bullish)' if price > ema200 else 'BELOW ❌ (macro bearish)')
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
EMA200 : {ema200_str} → Price {ema200_arrow}
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

━━━ STEP 7: SL/TP RULES (MANDATORY — violating these = output WAIT) ━━━
ATR(14): {atr} | 0.8×ATR={round(atr*0.8,6)} | 2.5×ATR={round(atr*2.5,6)} | 3.0×ATR={round(atr*3.0,6)}

SL PLACEMENT:
→ LONG:  SL just below the most recent swing low or bullish OB + 0.3×ATR buffer ({round(atr*0.3,6)})
         Max SL distance from entry = 3.0×ATR = {round(atr*3.0,6)} — output WAIT if SL must be wider
→ SHORT: SL just above the most recent swing high or bearish OB + 0.3×ATR buffer ({round(atr*0.3,6)})
         Max SL distance from entry = 3.0×ATR = {round(atr*3.0,6)} — output WAIT if SL must be wider

TP1 PLACEMENT — THE SINGLE MOST IMPORTANT RULE:
→ LONG:  TP1 MUST equal the nearest BSL (buy-side liquidity) between {round(atr*0.8,6)} and {round(atr*2.5,6)} above entry
         Available BSL levels (stops clustered above recent highs): {', '.join(str(round(p, 6)) for p in liq['bsl'])}
→ SHORT: TP1 MUST equal the nearest SSL (sell-side liquidity) between {round(atr*0.8,6)} and {round(atr*2.5,6)} below entry
         Available SSL levels (stops clustered below recent lows): {', '.join(str(round(p, 6)) for p in liq['ssl'])}
→ If NO qualifying BSL/SSL exists in that ATR range → output WAIT. Do NOT invent a TP.
→ NEVER use ATR math to set TP1. Only use the real liquidity levels listed above.

TP2: next liquidity pool beyond TP1, or the nearest 4h structural high/low
TP3 (runner): only if 4h trend agrees — target {round(atr*3.5,6)} to {round(atr*5.0,6)} from entry

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
   - 4h bias is NOT BEARISH (4H veto — BULLISH or NEUTRAL only)
   - RSI < 75 (not overbought) and ideally between 38-62 for pullback entry
   - MACD bullish or turning bullish (not opposing)
   - Volume HIGH or NORMAL (not VERY LOW)
   - Price at or near a BULLISH OB or BULL FVG (not inside a supply zone)
   - At least one: bullish OB, bull FVG, or bullish candle pattern confirming
   - Final confidence ≥ 72

✅ OUTPUT SHORT ONLY IF ALL TRUE:
   - TF overall = BEARISH (≥ 2/3 agree)
   - 4h bias is NOT BULLISH (4H veto — BEARISH or NEUTRAL only)
   - RSI > 25 (not oversold) and ideally between 38-62 for pullback entry
   - MACD bearish or turning bearish (not opposing)
   - Volume HIGH or NORMAL (not VERY LOW)
   - Price at or near a BEARISH OB or BEAR FVG (not inside a demand zone)
   - At least one: bearish OB, bear FVG, or bearish candle pattern confirming
   - Final confidence ≥ 72

⛔ OUTPUT WAIT IF ANY:
   - TF overall = MIXED
   - 4h bias CONTRADICTS the overall direction (e.g., overall BULLISH but 4h BEARISH)
   - Volume = VERY LOW
   - RSI > 75 and signal is LONG (chasing overbought)
   - RSI < 25 and signal is SHORT (chasing oversold)
   - Price inside a SUPPLY OB/FVG and signal is LONG
   - Price inside a DEMAND OB/FVG and signal is SHORT
   - MACD opposes the trade direction
   - Doji candle with no other confluence
   - Confidence < 72
   - You are unsure — WHEN IN DOUBT, WAIT. A skipped trade is NOT a lost trade.

Remember: The best traders miss many trades. They only take the perfect setup.
Every WAIT decision you make is protecting capital.

Respond ONLY with this exact JSON (no markdown, no preamble, no explanation):
{{"action":"LONG or SHORT or WAIT","confidence":0-100,"entry":{price},"sl":0.0,"tp1":0.0,"tp2":0.0,"tp3":0.0,"tp1_anchor":"BSL@<exact_price> or SSL@<exact_price> — MANDATORY for LONG/SHORT, empty string for WAIT","rr":"1:X.X","setup_type":"exact SMC setup name e.g. Bullish OB+FVG+Engulfing / Liquidity Sweep+BOS+Retest / WAIT-mixed-TF","session":"{self._trading_session()}","candle_pattern":"{candle_pattern}","timeframe_bias":"{tf_bias['overall']}","sentiment":"{fng['label']}","volume_signal":"{volume_signal}","reason":"3 sentences: (1) SMC structure and why this entry (2) Volume + candlestick + RSI confluence (3) Key risk factor and what would invalidate this trade","key_level":"most critical price level to watch","invalidation":"exact price that completely invalidates this setup — where you would cut the loss"}}'''

    # ─────────────────────────────────────────────────────────────────────────
    # BATCH SIGNAL  (one Claude call for N coins — replaces N individual calls)
    # ─────────────────────────────────────────────────────────────────────────

    def build_compact_summary(self, data: dict) -> str:
        """~250-token compact block for one coin, used inside the batch prompt."""
        price   = data['price']
        atr     = data['atr']
        tf_bias = data['timeframe_bias']
        per_tf  = tf_bias.get('per_tf', {})
        fng     = data['fear_greed']
        bos     = data.get('bos_choch', {'type': 'NONE', 'level': 0, 'confirmed': False})
        regime  = data.get('regime', 'RANGING')
        funding = data.get('funding_rate', {})
        obs     = data.get('order_blocks', [])
        fvgs    = data.get('fvgs', [])
        liq     = data.get('liquidity', {'bsl': [], 'ssl': []})
        ema200  = data.get('ema200')

        ob_str  = ', '.join(
            f"{o['type']}@{o['price']:.4f}(R:{o.get('respect_count', 0)})" for o in obs
        ) or 'NONE'
        fvg_str = ', '.join(
            f"{f['type']} {f['bot']:.4f}-{f['top']:.4f}({f.get('partial_fill_pct', 0):.0f}%fill)"
            for f in fvgs
        ) or 'NONE'
        macd_dir = 'BULL' if data['macd'] > data['macd_sig'] else 'BEAR'
        bb_pos   = ('NEAR_UPPER' if price > data['bb_upper'] * 0.998 else
                    'NEAR_LOWER' if price < data['bb_lower'] * 1.002 else 'MID')

        return (
            f"── {data['symbol']} | pre-score={data.get('pre_score', 0)}/100 ──\n"
            f"TF: 15m={per_tf.get('15m','?')} 1h={per_tf.get('1h','?')} 4h={per_tf.get('4h','?')}"
            f" → {tf_bias['overall']} {tf_bias['agreement']}/3\n"
            f"Regime={regime} | BOS/CHoCH={bos['type']}@{bos['level']} "
            f"({'CONFIRMED' if bos['confirmed'] else 'UNCONFIRMED'})\n"
            f"Price={price} ATR={atr:.4f} | EMA20={data['ema20']} EMA50={data['ema50']}"
            f" EMA200={ema200 if ema200 else 'N/A'}\n"
            f"RSI={data['rsi']} | RSI_Div={data.get('rsi_divergence','NONE')[:25]}"
            f" | MACD={macd_dir} | BB={bb_pos}\n"
            f"Vol={data.get('volume_signal','NORMAL')} trend={data.get('volume_trend','FLAT')}\n"
            f"Pattern={data.get('candle_pattern','None')[:40]}\n"
            f"OBs={ob_str} | FVGs={fvg_str}\n"
            f"BSL={[round(x, 4) for x in liq.get('bsl', [])[:2]]}"
            f" SSL={[round(x, 4) for x in liq.get('ssl', [])[:2]]}\n"
            f"News={data.get('news_sentiment','N/A')[:50]}"
            f" | F&G={fng.get('value', 50)}/{fng.get('label', '')}\n"
            f"Funding={funding.get('label', 'Neutral')[:40]}\n"
            f"SL_max_dist={round(atr*3.0,4)} from entry (0.3xATR buf={round(atr*0.3,4)})\n"
            f"TP1_range={round(atr*0.8,4)}-{round(atr*2.5,4)} | tp1_anchor REQUIRED (BSL for LONG, SSL for SHORT)"
        )

    def get_batch_signals(self, data_list: list) -> list:
        """
        Analyze multiple pre-filtered coins in ONE Claude call.
        Replaces N individual get_signal() calls — massive token saving.
        Returns a list of signal dicts (one per input coin).
        """
        if not data_list:
            return []

        # Regime filter: remove low-volatility coins before LLM call
        filtered, skipped = [], []
        for d in data_list:
            adx = d.get('adx', 0)
            bbp = d.get('bb_width_percentile', 50)
            if adx < 18 or bbp < 25:
                skipped.append({'action': 'WAIT', 'symbol': d.get('symbol', ''),
                                'confidence': 0, 'regime_skip': True,
                                'reason': f'REGIME_SKIP: ADX={adx:.1f} BB%={bbp}'})
            else:
                filtered.append(d)
        if not filtered:
            return skipped

        data_list = filtered
        n        = len(data_list)
        session  = self._trading_session()
        summaries = '\n\n'.join(self.build_compact_summary(d) for d in data_list)

        prompt = f'''You are a professional institutional crypto trader (15 years, SMC/ICT specialist).
Analyze {n} setups in ONE pass. Goal: protect capital. Only trade the 1-2 BEST setups.

━━ RULES (apply to every coin) ━━
LONG: TF_overall=BULLISH ≥2/3 | RSI<70 | volume≠VERY LOW | OB/FVG/CHoCH present | confidence≥72
SHORT: TF_overall=BEARISH ≥2/3 | RSI>30 | volume≠VERY LOW | OB/FVG/CHoCH present | confidence≥72
WAIT: TF=MIXED | VERY LOW volume | RSI>78+LONG | RSI<22+SHORT | no confluence | confidence<72
Priority: CHoCH > BOS > OB+FVG > OB alone | High respect_count = stronger zone
Unfilled FVG = price magnet. Min R:R 1:2.5. SL beyond OB/swing (max 3.0×ATR).
TP1=nearest BSL (LONG)/SSL (SHORT) in 0.8-2.5×ATR — tp1_anchor field MANDATORY | TP2=next pool | TP3=3.5-5×ATR | Session: {session}

━━ {n} SETUPS ━━

{summaries}

━━ OUTPUT ━━
Return exactly {n} objects (one per coin), sorted by confidence desc. WAIT all weak setups.
ONLY JSON array, no markdown, no extra text:
[{{"symbol":"X/USDT","action":"LONG|SHORT|WAIT","confidence":0-100,"entry":0.0,"sl":0.0,"tp1":0.0,"tp2":0.0,"tp3":0.0,"tp1_anchor":"BSL@<price> or SSL@<price> — MANDATORY for LONG/SHORT","rr":"1:X.X","setup_type":"setup name or WAIT-reason","reason":"why entry + key risk (2 sentences)","key_level":"critical level","invalidation":"exact invalidation price"}}]'''

        try:
            resp = self.client.messages.create(
                model=self.model_fast,   # sonnet — fast + cheap for scanning
                max_tokens=300 * n + 200,
                messages=[{'role': 'user', 'content': prompt}],
            )
            text = next(b.text for b in resp.content if b.type == 'text').strip()
            text = text.replace('```json', '').replace('```', '').strip()

            raw = json.loads(text)
            if not isinstance(raw, list):
                print('[Claude] Batch: response was not a JSON array')
                return []

            by_sym = {d['symbol']: d for d in data_list}
            result = []
            for sig in raw:
                sym  = sig.get('symbol', '')
                orig = by_sym.get(sym, {})
                sig.update({
                    'symbol':         sym,
                    'price':          orig.get('price', sig.get('entry', 0)),
                    'timestamp':      orig.get('timestamp', ''),
                    'pre_score':      orig.get('pre_score', 0),
                    'timeframe_bias': orig.get('timeframe_bias', {}).get('overall', 'MIXED'),
                    'volume_signal':  orig.get('volume_signal', 'NORMAL'),
                    'candle_pattern': orig.get('candle_pattern', ''),
                    'bos_choch':      orig.get('bos_choch', {'type': 'NONE', 'level': 0, 'confirmed': False}),
                    'regime':         orig.get('regime', 'RANGING'),
                    'sentiment':      orig.get('fear_greed', {}).get('label', 'Neutral'),
                    'session':        session,
                })
                # Confidence gate
                if sig.get('action') in ('LONG', 'SHORT') and sig.get('confidence', 0) < 72:
                    sig['action'] = 'WAIT'
                    sig['reason'] = (f"Confidence {sig['confidence']}% below 72. "
                                     + sig.get('reason', ''))

                sig['atr'] = orig.get('atr', 0)

                # tp1_anchor gate — TP1 must reference a real BSL/SSL level
                if sig.get('action') in ('LONG', 'SHORT'):
                    action_b   = sig['action']
                    tp1_anchor = sig.get('tp1_anchor', '').strip()
                    if not tp1_anchor:
                        sig['action'] = 'WAIT'
                        sig['reason'] = ('TP1_REJECT: tp1_anchor missing. '
                                         + sig.get('reason', ''))
                    else:
                        tp1_b  = sig.get('tp1', 0)
                        liq_b  = orig.get('liquidity', {'bsl': [], 'ssl': []})
                        target = liq_b.get('bsl', []) if action_b == 'LONG' else liq_b.get('ssl', [])
                        if tp1_b and target:
                            tol = tp1_b * 0.001
                            if not any(abs(tp1_b - lvl) <= tol for lvl in target):
                                lev_str = [round(l, 4) for l in target]
                                sig['action'] = 'WAIT'
                                sig['reason'] = (
                                    f'TP1_REJECT: TP1={tp1_b} not within 0.1% of '
                                    f'{"BSL" if action_b == "LONG" else "SSL"} {lev_str}. '
                                    + sig.get('reason', ''))

                result.append(sig)

            print(f'[Claude] Batch: {n} coins → 1 call → '
                  f"{sum(1 for s in result if s.get('action') in ('LONG','SHORT'))} signal(s)"
                  f" | {len(skipped)} regime-skipped")
            return skipped + result

        except Exception as e:
            print(f'[Claude] Batch error: {e}')
            return skipped

    # ─────────────────────────────────────────────────────────────────────────
    # GET SIGNAL  (kept for manual analyze / webhook — single-coin deep analysis)
    # ─────────────────────────────────────────────────────────────────────────
    def get_signal(self, market_data: dict, deep: bool = False) -> Optional[Dict]:
        try:
            # ── Phase 4: Regime filter — skip LLM entirely in choppy/low-vol markets
            adx = market_data.get('adx', 0)
            bbp = market_data.get('bb_width_percentile', 50)
            atp = market_data.get('atr_percentile', 50)
            sym = market_data.get('symbol', '')
            min_conf = 72

            if adx < 18 or bbp < 25:
                reason = f'Low-volatility regime (ADX={adx:.1f}, BB%ile={bbp:.0f}). Skipping LLM call.'
                print(f'[REGIME_SKIP] {sym} — {reason}')
                return {'action': 'WAIT', 'reason': reason, 'symbol': sym,
                        'confidence': 0, 'regime_skip': True}

            if atp > 95:
                min_conf = min_conf + 10   # extreme vol needs higher conviction

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
            signal['atr']       = market_data.get('atr', 0)

            # Confidence gate (min_conf raised to 82 during extreme ATR)
            if signal.get('action') in ('LONG', 'SHORT') and signal.get('confidence', 0) < min_conf:
                signal['action'] = 'WAIT'
                signal['reason'] = (f"Confidence {signal['confidence']}% below minimum {min_conf}. "
                                    + signal.get('reason', ''))

            # tp1_anchor gate — TP1 must reference a real BSL/SSL level
            if signal.get('action') in ('LONG', 'SHORT'):
                action     = signal['action']
                tp1_anchor = signal.get('tp1_anchor', '').strip()
                if not tp1_anchor:
                    signal['action'] = 'WAIT'
                    signal['reason'] = ('TP1_REJECT: tp1_anchor missing — TP1 not anchored to a '
                                        'real BSL/SSL level. ' + signal.get('reason', ''))
                else:
                    tp1 = signal.get('tp1', 0)
                    liq = market_data.get('liquidity', {'bsl': [], 'ssl': []})
                    target = liq.get('bsl', []) if action == 'LONG' else liq.get('ssl', [])
                    if tp1 and target:
                        tol = tp1 * 0.001  # ±0.1%
                        if not any(abs(tp1 - lvl) <= tol for lvl in target):
                            lev_str = [round(l, 4) for l in target]
                            signal['action'] = 'WAIT'
                            signal['reason'] = (
                                f'TP1_REJECT: TP1={tp1} not within 0.1% of any '
                                f'{"BSL" if action == "LONG" else "SSL"} level {lev_str}. '
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