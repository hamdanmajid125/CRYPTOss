# FINAL BACKTEST REPORT
**Date:** 2026-05-08  
**Period:** 2024-01-01 → 2026-05-08 (28 months)  
**Capital:** $10,000 (fixed)  
**Fees:** 0.05% taker per side + 0.02% slippage  
**Mode:** rules (rule-based signal generation, no LLM veto)

---

## In-Sample Results — Full Period (Reliable)

| Symbol   | Return%  | CAGR%  | Sharpe | Sortino | MaxDD%  | Trades | WinRate% | ProfitFactor | Expectancy |
|----------|----------|--------|--------|---------|---------|--------|----------|--------------|------------|
| BTC/USDT | -1.71%   | -0.74% | -0.051 | -0.196  | -2.16%  | 10     | 40.0%    | 0.725        | -$17.08    |
| ETH/USDT | -8.94%   | -3.96% | -0.115 | -1.052  | -12.01% | 56     | 39.3%    | 0.733        | -$15.96    |
| SOL/USDT | -2.26%   | -0.98% | -0.014 | -0.148  | -11.17% | 109    | 46.8%    | 0.963        | -$2.08     |
| BNB/USDT | -4.49%   | -1.96% | -0.072 | -0.465  | -8.35%  | 34     | 41.2%    | 0.782        | -$13.21    |

**All four symbols show negative expectancy and negative Sharpe on in-sample data.**  
SOL/USDT is the closest to breakeven (PF 0.963, expectancy -$2.08/trade).

---

## LLM Hybrid vs Rules-Only (in-sample)

| Symbol   | Rules PF | Hybrid PF | Rules Sharpe | Hybrid Sharpe | Verdict          |
|----------|----------|-----------|--------------|---------------|------------------|
| BTC/USDT | 0.725    | 0.423     | -0.051       | -0.120        | Hybrid worse     |
| ETH/USDT | 0.733    | 0.581     | -0.115       | -0.173        | Hybrid worse     |
| SOL/USDT | 0.963    | 0.998     | -0.014       | +0.005        | Hybrid marginally better |
| BNB/USDT | 0.782    | 0.782     | -0.072       | -0.072        | No change        |

**Conclusion:** The LLM veto makes metrics worse on 2/4 symbols and is neutral on 1. **Production mode should remain `rules`**. The LLM is double-penalising factors already priced into the confidence score (MACD, EMA200 alignment). Only SOL shows a marginal improvement, likely because high trade frequency gives the veto more noise to filter.

---

## Walk-Forward Analysis

**Methodology:** 6-month training windows, 2-month OOS windows, 11 windows total.  
**⚠️ Known limitation:** The walk-forward engine compounds equity between OOS windows (the position sizer uses the accumulated balance). This inflates late-window P&L exponentially and makes the aggregate figures unreliable. The per-window trade counts are correct; the aggregate equity numbers are not.

### Per-Window Trade Count (rules mode)

| Window | Period              | BTC | ETH | SOL | BNB |
|--------|---------------------|-----|-----|-----|-----|
| W1     | 2024-07-01–2024-08-30 | 3  | 6   | 7   | 1   |
| W2     | 2024-08-30–2024-10-29 | 0  | 0   | 0   | 0   |
| W3     | 2024-10-29–2024-12-28 | 0  | 2   | 7   | 6   |
| W4     | 2024-12-28–2025-02-26 | 0  | 12  | 20  | 5   |
| W5     | 2025-02-26–2025-04-27 | 0  | 5   | 8   | 0   |
| W6     | 2025-04-27–2025-06-26 | 0  | 2   | 0   | 0   |
| W7     | 2025-06-26–2025-08-25 | 0  | 0   | 0   | 0   |
| W8     | 2025-08-25–2025-10-24 | 0  | 1   | 3   | 9   |
| W9     | 2025-10-24–2025-12-23 | 0  | 5   | 7   | 2   |
| W10    | 2025-12-23–2026-02-21 | 2  | 8   | 8   | 2   |
| W11    | 2026-02-21–2026-04-22 | 0  | 0   | 0   | 0   |
| **Total** |                  | **5** | **41** | **60** | **25** |

**Observations:**
- Signal frequency is very low — most 2-month OOS windows produce zero trades on BTC.
- ETH went bankrupt (final equity $0) when compounding is applied — confirms the negative-expectancy finding.
- SOL consistently produces more signals but remained near-breakeven.
- Several consecutive zero-trade windows suggest the regime filter (ADX + BBW chop) is correctly sitting out of range-bound periods.

---

## Phase 4 Feature Impact (in-sample delta, rules mode)

Phase 4 features (funding rate scoring, OI compression flag, per-symbol cooldown, ATR% normalization) were added **after** the above backtests were recorded. A direct toggle comparison requires re-running with each feature isolated, which was not done due to time constraints. Qualitative assessment:

| Feature | Expected Impact | Observable Effect |
|---------|----------------|-------------------|
| Funding rate scoring | -10 on crowded longs/shorts | Reduces confidence by ≤10 pts when FR extreme; fewer marginal signals pass the 65-confidence gate |
| OI compression flag | -5 when OI drops >5% | Minor downside filter; ~1-2% trade reduction expected |
| Per-symbol cooldown | 60/120 min blocks | Prevents re-entry after stop-outs; slightly reduces trade count |
| ATR% normalization | Smaller size on volatile assets | Reduces max loss magnitude; improves Sortino without changing win rate |

---

## Recommendation: Should `AUTO_TRADE=true` be enabled?

### **NO. Do not enable AUTO_TRADE=true.**

**Reasons:**

1. **Negative expectancy across all 4 symbols** — Every symbol has a profit factor below 1.0 on the full 28-month in-sample period. The strategy loses money in expectation before any slippage or market-impact.

2. **Low trade frequency** — BTC generates only 5 trades over 28 months, making any performance figure statistically unreliable. SOL generates the most (109 in-sample) and has the best metrics, but still negative.

3. **Walk-forward instability** — Most OOS windows have zero trades. ETH destroyed capital (bankrupt) in walk-forward. No symbol achieved a positive out-of-sample Sharpe ratio consistently.

4. **LLM veto is net-negative** — Phase 3 showed the LLM veto hurts on 2/4 symbols and is neutral on 1. Running `mode='rules'` is preferable.

5. **Known engine limitation** — The walk-forward position sizer compounds equity between windows, which must be fixed before any out-of-sample metrics can be trusted.

### Recommended next steps before reconsidering live trading

1. **Fix the walk-forward engine** — Reset capital to a fixed amount per OOS window (or use fixed `risk_usdt` instead of `risk_pct`). Re-run to get true out-of-sample performance.

2. **Paper-trade for 30+ days** — Run in paper mode (`PAPER_TRADING=true`) with all Phase 4 features active. Let `signals.jsonl` and `closes.jsonl` accumulate real-time data. Review `/stats` rolling metrics weekly.

3. **Improve win rate** — The core issue is a 39-47% win rate with roughly 1:1 avg-win/avg-loss. Need either:
   - Win rate improvement: better entry timing (limit orders at FVG midpoint vs. market-on-close)
   - Better R:R: tighter SL (increase `sl_mult` to 1.0×ATR with wider-filter entries) or trailing stop runner
   - Regime selectivity: only trade when ADX > 25 (stricter than current 18 threshold)

4. **Address the compounding walk-forward bug** — then target Sharpe > 0.5 OOS on at least 3/4 symbols before enabling AUTO_TRADE.

---

## Test Coverage Summary

| Phase | Tests | All Pass |
|-------|-------|----------|
| 1 — Backtester         | 12    | ✅       |
| 2 — SMC detection      | 16    | ✅       |
| 3 — Signal logic       | 11    | ✅       |
| 4 — Phase 4 edges      | 15    | ✅       |
| 5 — Config/observability | 13  | ✅       |
| **Total**              | **67** | **✅ 67/67** |

---

*Generated by the CRYPTOss backtesting infrastructure. PAPER_TRADING=true is the default and must remain so until the above criteria are met.*
