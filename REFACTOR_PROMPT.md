# CRYPTOss — Refactor & Profitability Hardening Prompt

> Paste this entire file into Claude Code as your initial message. Work through it phase by phase, in order. Do not skip ahead.

---

## Mission

You are working on a crypto trading bot located in this repository. The bot generates LONG/SHORT/WAIT signals on BTC, ETH, SOL, BNB using indicators + Smart Money Concepts (SMC) + an LLM decision step. A code review found that the infrastructure is solid but **the strategy is unproven and several detection routines are buggy**. Your job is to fix the bugs, restructure the decision flow, add missing edges, and — most importantly — **prove profitability with a backtest before anything else changes in production**.

## Working agreement

1. **Work phase by phase, in the order given.** After each phase, run the tests, summarize results in chat, and wait for my "go" before starting the next phase.
2. **Do not rewrite code that is not on the fix list.** The FastAPI server, WebSocket dashboard contract, and `RiskManager` public interface must keep working.
3. **Every new module gets a unit test.** Use `pytest`. Tests live in `tests/`.
4. **Use the libraries already in the project** (`ccxt`, `pandas`, `pandas_ta_classic`, `anthropic`, `fastapi`). Do not add heavy frameworks unless explicitly asked.
5. **Commit after each phase** with a clear message: `phase-1: backtester`, `phase-2: indicator fixes`, etc.
6. **Never push live trading config.** `.env` must stay gitignored. Default `PAPER_TRADING=true`.

## Repo orientation

- `main.py` — FastAPI server, scan endpoints, auto-trade execution loop
- `data_feed.py` — OHLCV fetcher + indicator computation + SMC detection (FVG, OB, liquidity, MTF bias)
- `claude_agent.py` — Builds prompt, calls Claude, parses JSON signal
- `risk_manager.py` — Position sizing, daily loss limit, drawdown halt, correlation filter
- `weex_client.py` — Order routing (paper/live)
- `trading_dashboard.html` — Frontend (do not touch unless I ask)

---

## PHASE 1 — Build a backtester (CRITICAL, do this before any other change)

The bot has never been backtested. Until we have a backtest, every other "improvement" is guesswork.

### Tasks

1. Create `backtest/` directory with:
   - `backtest/engine.py` — event-driven backtester
   - `backtest/runner.py` — CLI entry point
   - `backtest/metrics.py` — performance metrics
   - `backtest/data_loader.py` — historical OHLCV loader (use `ccxt` with `since` parameter, cache to local Parquet files in `backtest/data/`)

2. The engine must:
   - Replay historical 1h candles bar by bar for a given symbol and date range
   - At each bar close, compute the *exact same* indicators and SMC features that `data_feed.py` produces (refactor `data_feed.py` so the indicator code is reusable from both live and backtest paths — pull pure functions out into `indicators.py`)
   - Compute the rule-based confidence score (`_score_confidence` from `claude_agent.py` — also extract this to a pure function in a new `signal_logic.py`)
   - Generate signals using a `--mode` flag: `rules` (no LLM, use pre_score directly), `llm` (call Claude), or `hybrid` (rules generate, LLM vetoes)
   - Apply the same `RiskManager` logic for position sizing and gating
   - Track open positions, hit SL/TP1/TP2/TP3 based on intra-bar high/low, scale out at TPs (50% TP1, 30% TP2, 20% TP3), move SL to breakeven after TP1
   - Subtract realistic fees (default 0.05% per side, taker) and slippage (default 0.02%) from every fill

3. `metrics.py` must report:
   - Total return %, CAGR
   - Sharpe ratio (annualized, daily-resampled returns)
   - Sortino ratio
   - Max drawdown %, max DD duration
   - Win rate, average win, average loss, expectancy per trade
   - Profit factor
   - Number of trades
   - Best/worst trade
   - Equity curve as CSV + matplotlib PNG

4. `runner.py` CLI:
   ```
   python -m backtest.runner --symbol BTC/USDT --start 2024-01-01 --end 2025-12-01 --mode rules --capital 10000
   ```
   Output: prints metrics table, saves equity curve to `backtest/results/{symbol}_{mode}_{start}_{end}.{csv,png}`

5. **Walk-forward validation:** add `--walk-forward` flag that splits the date range into 6-month train / 2-month test rolling windows, reports out-of-sample metrics separately.

### Acceptance criteria for Phase 1

- `pytest tests/test_backtester.py` passes (write at least 5 tests: bar replay correctness, fee deduction, SL/TP hit logic on synthetic data, scale-out math, equity curve continuity).
- I can run `python -m backtest.runner --symbol BTC/USDT --start 2024-01-01 --end 2025-12-01 --mode rules` and see a metrics summary.
- Run the same backtest in all three modes (`rules`, `llm`, `hybrid`) for BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT over Jan 2024 to today. Save results. **Report the metrics table to me. Do not proceed to Phase 2 until I confirm.**

---

## PHASE 2 — Fix the indicator and SMC detection bugs

These are correctness bugs in `data_feed.py`. They produce false features that the LLM and rule scorer rely on.

### 2.1 Fix `find_fvgs`

Current code only checks two candles. A real ICT Fair Value Gap is a 3-candle pattern where the middle candle's range leaves an imbalance between candle 1 and candle 3.

Replace with proper 3-candle logic:
- **Bullish FVG:** `candle[i+1].low > candle[i-1].high` (gap above), and the middle candle `i` is a strong displacement candle (body > 0.5 × ATR at that bar). The FVG zone is `[candle[i-1].high, candle[i+1].low]`.
- **Bearish FVG:** `candle[i+1].high < candle[i-1].low`, middle candle is strong displacement. Zone: `[candle[i+1].high, candle[i-1].low]`.
- Track **mitigation:** mark an FVG as `mitigated=True` if any later candle's wick has fully filled the zone. Only return unmitigated FVGs.
- Return at most the 5 most recent unmitigated FVGs.

### 2.2 Fix `find_order_blocks`

Current code finds momentum candles, not real OBs.

Replace with proper OB logic:
- A **bullish OB** is the last bearish candle (close < open) before a sequence of candles that breaks the most recent swing high (Break of Structure). The OB zone is the high-to-low range of that bearish candle.
- A **bearish OB** is the last bullish candle before a sequence that breaks the most recent swing low.
- Detect swing points using a 5-bar pivot (high is highest of 5 surrounding bars).
- Track **mitigation:** mark an OB as `mitigated=True` once price has traded back through 50% of its range. Only return unmitigated OBs.
- Return at most the 3 most recent unmitigated OBs per direction.

### 2.3 Fix `detect_rsi_divergence`

Current code compares overlapping windows and produces unreliable signals.

Replace with pivot-based divergence:
- Find local price highs/lows using 5-bar pivots over the last 50 bars.
- For **bearish divergence:** two consecutive price pivot highs where price-pivot-2 > price-pivot-1 but RSI-at-pivot-2 < RSI-at-pivot-1.
- For **bullish divergence:** two consecutive price pivot lows where price-pivot-2 < price-pivot-1 but RSI-at-pivot-2 > RSI-at-pivot-1.
- Only count divergences where the second pivot is within the last 5 bars (otherwise the signal is stale).

### 2.4 Fix `find_liquidity_levels`

Current code returns the top 3 highs and bottom 3 lows. Real liquidity sits at *equal* highs/lows where stops cluster.

Replace with cluster detection:
- Find all swing highs in the last 50 bars (5-bar pivots).
- Cluster swing highs that are within 0.3 × ATR of each other. A cluster of 2+ swing highs is a BSL pool.
- Same for swing lows → SSL pools.
- Return clusters with `{level, count, last_touch_bar}`.
- Track **swept:** mark a pool as swept if a later candle's wick traded through the level by more than 0.1 × ATR. Return only unswept pools.

### Acceptance criteria for Phase 2

- All four functions have unit tests with hand-crafted candle data verifying correct detection on known patterns and rejection of false patterns.
- Re-run the Phase 1 backtests in `rules` mode. **Compare metrics before and after Phase 2.** Report both. The detection fixes should change signal frequency and quality measurably.

---

## PHASE 3 — Restructure the decision flow

Currently the LLM has the final say on confidence. The LLM has no edge on numerical price prediction. Flip the architecture so rules generate signals and the LLM acts as a quality filter.

### Tasks

1. In `signal_logic.py` (extracted in Phase 1), make `generate_signal(market_data) -> dict` a pure function that:
   - Computes the rule-based confidence score (refactored from `_score_confidence`)
   - Computes entry, SL (1.5 × ATR from entry), TP1 (2.0 × ATR), TP2 (3.5 × ATR), TP3 (5.5 × ATR) using the existing ATR logic
   - Applies all hard gates: MTF agreement ≥ 2/3, EMA200 alignment, R:R ≥ 2.0 after fees
   - Returns `{action, confidence, entry, sl, tp1, tp2, tp3, rr, setup_type, ...}` or `{action: 'WAIT', reason: ...}`

2. Modify `claude_agent.py`:
   - Add a new method `veto_signal(rule_signal, market_data) -> dict` that takes a rule-generated signal and asks Claude **only** to approve, reject, or downgrade confidence — not to invent its own entry/SL/TP. The LLM gets the signal as context and outputs `{decision: APPROVE | REJECT | DOWNGRADE, confidence_adjustment: -20 to +10, reason: str}`.
   - Use the **structured outputs / tool_use** feature so JSON parsing cannot fail. Define a strict schema with `tools` and `tool_choice`. Use `claude-sonnet-4-6` for fast vetoes, `claude-opus-4-7` only for high-stakes/large-size trades.
   - Keep the old `get_signal` method but mark it deprecated.

3. In `main.py`, change the flow:
   - `_analyze_market` calls `signal_logic.generate_signal(data)` first
   - If `action != WAIT`, call `claude.veto_signal(signal, data)`
   - Apply the LLM's `confidence_adjustment` to the signal's confidence
   - If decision is `REJECT`, set action to `WAIT`
   - Then pass to `risk_manager.can_trade()`

### Acceptance criteria for Phase 3

- Existing endpoints still work and the dashboard still receives signals in the same JSON shape.
- Unit tests for `generate_signal` covering: bullish setup passes, bearish setup passes, MTF disagreement → WAIT, low R:R → WAIT, low pre-score → WAIT.
- Re-run backtests in `hybrid` mode (rules + LLM veto). Compare to pure `rules` mode. **The LLM veto must not make metrics worse.** If hybrid is worse than rules-only, report it and recommend dropping the LLM step.

---

## PHASE 4 — Add the missing edges

These are the features that materially affect crypto profitability and are currently absent.

### 4.1 Regime filter

Add to `signal_logic.generate_signal`:
- Compute ADX(14) and Bollinger Band Width (BBW = (BB_upper - BB_lower) / BB_middle).
- **Block all trades when** `ADX < 18` AND `BBW` is in the bottom 30% of its 100-bar rolling distribution. This is chop — SMC bleeds in chop.
- Allow trades when `ADX > 20` OR `BBW` is expanding (BBW now > 1.2 × BBW 10 bars ago).

### 4.2 Funding rate + open interest

Add `get_funding_rate(symbol)` and `get_open_interest(symbol)` to `data_feed.py` using Binance perp endpoints (`fapi/v1/fundingRate` and `fapi/v1/openInterest`).

Add to the rule scorer:
- Funding > +0.05% (8h) and signal is LONG → subtract 10 from confidence (longs are crowded).
- Funding < -0.05% and signal is SHORT → subtract 10 (shorts are crowded).
- Funding > +0.1% with bearish MTF bias → add 10 to SHORT confidence (long squeeze setup).
- OI rising > 5% in 4h with price flat → flag as "compression, expect breakout" in signal metadata.

### 4.3 Per-symbol cooldown

In `risk_manager.py`:
- Track `last_close_time[symbol]`.
- After any close, block new entries on that symbol for **60 minutes**.
- After a *stop-out* specifically, block for **120 minutes** OR until MTF bias flips, whichever comes first.

### 4.4 Per-symbol risk normalization by ATR%

In `RiskManager.position_size`:
- Compute `atr_pct = atr / entry`.
- If `atr_pct > 0.025` (high vol), scale `risk_usdt` down by `0.025 / atr_pct`.
- This prevents over-sizing on volatile symbols.

### 4.5 Fee-aware R:R gate

In `signal_logic.generate_signal`:
- Compute `effective_tp1 = tp1 - 2 * fee_pct * entry` for longs (subtract fees from reward).
- Compute `effective_sl = sl - 2 * fee_pct * entry` for longs (add fees to risk — sl is further away).
- R:R must be ≥ 2.0 *after* this adjustment.

### Acceptance criteria for Phase 4

- Unit tests for each filter.
- Backtest with all Phase 4 features enabled. Compare metrics to end-of-Phase-3. Report whether each individual feature helped or hurt by toggling them on/off via config flags.

---

## PHASE 5 — Production hardening

Only after Phase 1–4 show positive expectancy on out-of-sample data.

### Tasks

1. **Configuration:** Move all magic numbers (1.5 × ATR, 0.025 atr_pct threshold, 60-min cooldown, etc.) to `config.yaml`. Load via Pydantic settings.
2. **Observability:** Add structured logging (`structlog`). Log every signal decision with all input features as JSON to `signals.jsonl`. This becomes the dataset for future model improvements.
3. **Rolling expectancy:** Add a `rolling_metrics()` method to `RiskManager` that computes 30-day win rate, expectancy, and Sharpe from `signals.jsonl` + close events. Surface on `/stats`. **Auto-pause if 30-day expectancy turns negative.**
4. **Alerts:** Webhook (Telegram or Discord) on: trade open, trade close, daily loss limit hit, drawdown threshold hit, bot pause.
5. **Repo hygiene:** Confirm `.env`, `*.log`, `__pycache__/` are in `.gitignore`. Remove `server.log` and `trades.log` from git history if present (`git rm --cached`).

### Acceptance criteria for Phase 5

- `config.yaml` controls all tunables. Changing values does not require code edits.
- `signals.jsonl` logs are produced and parseable.
- Telegram or Discord webhook fires on a test trade.
- Final out-of-sample backtest report in `backtest/results/FINAL_REPORT.md` with: Sharpe, Sortino, max DD, profit factor per symbol, walk-forward stability table, and a recommendation on whether to enable `AUTO_TRADE=true` in production.

---

## What "done" looks like

- `pytest` passes with at least 30 unit tests across the new modules.
- Backtester runs on all 4 symbols across 2024–2025, all 3 modes, and produces an HTML or markdown report.
- Out-of-sample (walk-forward) Sharpe ratio is positive after fees and slippage on at least 3 of 4 symbols.
- The LLM step demonstrably either helps or has been removed.
- The bot runs in paper mode for at least 7 consecutive days with the new logic before any live capital is risked.

## What you must NOT do

- Do not enable live trading. `PAPER_TRADING=true` stays default.
- Do not remove the existing risk manager safeguards (daily loss limit, drawdown halt, correlation filter, consecutive-loss pause).
- Do not add new symbols beyond the existing 4 until the strategy is proven on those.
- Do not over-fit by tuning parameters per symbol on the same data you backtest on. Use walk-forward.
- Do not silently catch and swallow exceptions. Log them.

## Start

Begin with Phase 1. Read the four core files first (`main.py`, `data_feed.py`, `claude_agent.py`, `risk_manager.py`), then create the `backtest/` directory and `indicators.py`/`signal_logic.py` extractions. Stop and report when Phase 1 acceptance criteria are met.
