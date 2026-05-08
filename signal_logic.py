"""
Pure signal-generation logic — shared by claude_agent.py and the backtester.
No I/O, no exchange connections, no Claude API calls.
"""


def score_confidence(data: dict, tf_bias: dict, fng: dict) -> int:
    """Rule-based confidence scorer extracted from ClaudeAgent._score_confidence."""
    score = 50

    agreement = tf_bias.get('agreement', 0)
    if agreement == 3:    score += 20
    elif agreement == 2:  score += 8
    else:                 score -= 20

    vol = data.get('volume_signal', 'NORMAL')
    if 'VERY HIGH' in vol:  score += 15
    elif 'HIGH'    in vol:  score += 8
    elif 'VERY LOW' in vol: score -= 15
    elif 'LOW'     in vol:  score -= 8

    vt = data.get('volume_trend', 'FLAT')
    if 'INCREASING' in vt:  score += 5
    elif 'DECREASING' in vt: score -= 5

    fng_val = fng.get('value', 50)
    overall = tf_bias.get('overall', 'MIXED')
    if   fng_val < 20 and overall == 'BULLISH':  score += 12
    elif fng_val > 80 and overall == 'BEARISH':  score += 12
    elif fng_val < 35 and overall == 'BULLISH':  score += 6
    elif fng_val > 65 and overall == 'BEARISH':  score += 6

    rsi = data.get('rsi', 50)
    if   overall in ('BULLISH', 'BEARISH') and 38 <= rsi <= 62: score += 10
    elif rsi > 75 and overall == 'BULLISH':  score -= 15
    elif rsi < 25 and overall == 'BEARISH':  score -= 15

    rsi_div = data.get('rsi_divergence', 'NONE')
    if 'BULLISH' in rsi_div and overall == 'BULLISH': score += 12
    elif 'BEARISH' in rsi_div and overall == 'BEARISH': score += 12

    pattern = data.get('candle_pattern', '')
    high_conf = ['Bullish Engulfing', 'Bearish Engulfing', 'Morning Star',
                 'Evening Star', 'Bullish Marubozu', 'Bearish Marubozu']
    med_conf  = ['Hammer', 'Shooting Star']
    if any(p in pattern for p in high_conf): score += 12
    elif any(p in pattern for p in med_conf): score += 6
    elif 'Doji' in pattern: score -= 5

    fvgs  = data.get('fvgs', [])
    price = data.get('price', 0)
    in_bull_fvg = any(f['bot'] <= price <= f['top'] for f in fvgs if f.get('type') in ('bullish', 'bull'))
    in_bear_fvg = any(f['bot'] <= price <= f['top'] for f in fvgs if f.get('type') in ('bearish', 'bear'))
    if overall == 'BULLISH':
        if in_bull_fvg: score += 10
        if in_bear_fvg: score -= 12
    elif overall == 'BEARISH':
        if in_bear_fvg: score += 10
        if in_bull_fvg: score -= 12

    obs = data.get('order_blocks', [])
    for o in obs:
        top = o.get('top', o.get('price', price) * 1.001)
        bot = o.get('bot', o.get('price', price) * 0.999)
        if not (bot <= price <= top):
            continue
        if overall == 'BULLISH' and o['type'] == 'bullish':   score += 12
        elif overall == 'BEARISH' and o['type'] == 'bearish': score += 12
        elif overall == 'BULLISH' and o['type'] == 'bearish': score -= 12
        elif overall == 'BEARISH' and o['type'] == 'bullish': score -= 12

    per_tf = tf_bias.get('per_tf', {})
    if per_tf.get('4h') == 'BEARISH' and overall == 'BULLISH': score -= 25
    elif per_tf.get('4h') == 'BULLISH' and overall == 'BEARISH': score -= 25

    ema200 = data.get('ema200')
    if ema200 is not None:
        if   overall == 'BULLISH' and price > ema200: score += 5
        elif overall == 'BULLISH' and price < ema200: score -= 10
        elif overall == 'BEARISH' and price < ema200: score += 5
        elif overall == 'BEARISH' and price > ema200: score -= 10

    macd     = data.get('macd', 0)
    macd_sig = data.get('macd_sig', 0)
    if   overall == 'BULLISH' and macd > macd_sig:   score += 5
    elif overall == 'BEARISH' and macd < macd_sig:   score += 5
    elif overall == 'BULLISH' and macd < macd_sig:   score -= 8
    elif overall == 'BEARISH' and macd > macd_sig:   score -= 8

    return max(0, min(100, score))


def generate_signal(
    market_data: dict,
    fee_pct: float = 0.0005,
    min_rr: float = 2.0,
    min_confidence: int = 65,
) -> dict:
    """
    Pure rule-based signal generator.

    Returns a signal dict:
        {action, confidence, entry, sl, tp1, tp2, tp3, rr, setup_type, atr}
    or a WAIT dict:
        {action: 'WAIT', reason: str}
    """
    tf_bias    = market_data.get('timeframe_bias', {})
    fng        = market_data.get('fear_greed', {'value': 50, 'label': 'Neutral'})
    overall    = tf_bias.get('overall', 'MIXED')
    agreement  = tf_bias.get('agreement', 0)
    per_tf     = tf_bias.get('per_tf', {})

    if overall == 'MIXED' or agreement < 2:
        return {'action': 'WAIT', 'reason': f'MTF MIXED or agreement only {agreement}/3'}

    if overall == 'BULLISH' and per_tf.get('4h') == 'BEARISH':
        return {'action': 'WAIT', 'reason': '4H veto: 4H BEARISH kills BULLISH bias'}
    if overall == 'BEARISH' and per_tf.get('4h') == 'BULLISH':
        return {'action': 'WAIT', 'reason': '4H veto: 4H BULLISH kills BEARISH bias'}

    # Regime filter: ADX < 18 AND BB percentile < 30 → chop
    adx    = market_data.get('adx', 25)
    bb_pct = market_data.get('bb_width_percentile', 50)
    if (adx is not None and not _is_nan(adx) and adx < 18
            and not _is_nan(bb_pct) and bb_pct < 30):
        return {'action': 'WAIT', 'reason': f'Regime: chop (ADX={adx:.1f} BB%={bb_pct})'}

    # Volume hard gate
    if 'VERY LOW' in market_data.get('volume_signal', 'NORMAL'):
        return {'action': 'WAIT', 'reason': 'Volume VERY LOW'}

    action = 'LONG' if overall == 'BULLISH' else 'SHORT'

    price = float(market_data.get('price', 0) or 0)
    atr   = float(market_data.get('atr',   0) or 0)
    if price <= 0 or atr <= 0:
        return {'action': 'WAIT', 'reason': 'price or ATR is zero'}

    if action == 'LONG':
        entry = price
        sl    = entry - 1.5 * atr
        tp1   = entry + 2.0 * atr
        tp2   = entry + 3.5 * atr
        tp3   = entry + 5.5 * atr
        eff_reward = (tp1 - entry) - 2 * fee_pct * entry
        eff_risk   = (entry - sl) + 2 * fee_pct * entry
    else:
        entry = price
        sl    = entry + 1.5 * atr
        tp1   = entry - 2.0 * atr
        tp2   = entry - 3.5 * atr
        tp3   = entry - 5.5 * atr
        eff_reward = (entry - tp1) - 2 * fee_pct * entry
        eff_risk   = (sl - entry) + 2 * fee_pct * entry

    rr = eff_reward / eff_risk if eff_risk > 0 else 0.0
    if rr < min_rr:
        return {'action': 'WAIT', 'reason': f'R:R {rr:.2f} < {min_rr} after fees'}

    confidence = score_confidence(market_data, tf_bias, fng)
    if confidence < min_confidence:
        return {'action': 'WAIT', 'reason': f'Confidence {confidence} < {min_confidence}'}

    return {
        'action':     action,
        'confidence': confidence,
        'entry':      entry,
        'sl':         sl,
        'tp1':        tp1,
        'tp2':        tp2,
        'tp3':        tp3,
        'rr':         f'1:{rr:.1f}',
        'rr_float':   round(rr, 2),
        'setup_type': f'rule_{overall.lower()}',
        'atr':        atr,
    }


def _is_nan(v) -> bool:
    try:
        import math
        return math.isnan(float(v))
    except Exception:
        return False
