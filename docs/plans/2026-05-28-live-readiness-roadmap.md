# Live Readiness Roadmap Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make BAT safe enough to progress toward real BTCUSDT trading through staged live-readiness gates, extreme canary risk limits, model admission checks, explainability, data quality controls, and operational monitoring.

**Architecture:** Add safety layers around the existing Analyst/TUI/Broker flow instead of letting model signals call `broker.buy()` / `broker.sell()` directly. The system must progress through `OFF -> PAPER -> SHADOW -> CANARY -> LIVE_LIMITED`, with `OFF/PAPER` as the default and `CANARY` capped at extremely low exposure.

**Tech Stack:** Python 3.13, unittest, pandas/numpy, scikit-learn HistGB, Binance Spot SDK, Textual TUI, uv.

---

## Non-Negotiable Live Criteria

- Scope starts with `BTCUSDT` on `15m` only.
- Default remains PAPER/TESTNET. Real orders require explicit state, explicit config, and passing gates.
- First real-money mode is `CANARY`, not unrestricted live trading.
- CANARY limits: max 1% account equity per order, max 0.5% daily realized/unrealized loss, immediate downgrade to PAPER/OFF on abnormal data, API, model, or order state.
- No model can trade live unless it passes walk-forward, paper/shadow, explainability, and data-quality admission gates.
- Any failed gate returns a machine-readable reason and blocks order placement.

---

### Task 1: Live State And Risk Config Foundation

**Files:**
- Create: `src/bat/risk/__init__.py`
- Create: `src/bat/risk/live_state.py`
- Modify: `src/bat/research_config.py`
- Test: `tests/test_live_state.py`

**Step 1: Write failing tests**

Create tests for:
- default state is `PAPER`
- valid state transitions: `OFF -> PAPER -> SHADOW -> CANARY -> LIVE_LIMITED`
- invalid transition `PAPER -> LIVE_LIMITED` raises
- `RiskConfig` has canary defaults: `max_order_equity_fraction=0.01`, `max_daily_loss_fraction=0.005`, `paper_only=True`, `trading_enabled=False`

Run:
`PYTHONPATH=src uv run python -m unittest tests.test_live_state`

Expected: FAIL because module/config fields do not exist.

**Step 2: Implement minimal state model**

Add:
- `LiveState` enum-like string constants or `Enum`
- `validate_transition(current, requested)`
- `LiveStateError`

Extend `RiskConfig` with:
- `live_state: str = "PAPER"`
- `max_order_equity_fraction: float = 0.01`
- `max_daily_loss_fraction: float = 0.005`
- `max_consecutive_losses: int = 2`
- `cooldown_minutes: int = 60`

**Step 3: Verify**

Run:
`PYTHONPATH=src uv run python -m unittest tests.test_live_state`

Expected: PASS.

---

### Task 2: Pre-Trade Guard For Real Orders

**Files:**
- Create: `src/bat/risk/live_guard.py`
- Test: `tests/test_live_guard.py`

**Step 1: Write failing tests**

Test `LiveGuard.evaluate_order_request(...)` blocks when:
- state is `OFF`, `PAPER`, or `SHADOW`
- `paper_only=True`
- symbol is not `BTCUSDT`
- confidence is below threshold
- order quote amount exceeds 1% equity
- daily loss exceeds 0.5%
- data freshness is stale

Test it allows only when:
- state is `CANARY`
- `paper_only=False`
- `trading_enabled=True`
- symbol is `BTCUSDT`
- data/model/risk checks pass
- quote amount is capped to 1% equity

Run:
`PYTHONPATH=src uv run python -m unittest tests.test_live_guard`

Expected: FAIL because guard does not exist.

**Step 2: Implement guard result types**

Add small dataclasses:
- `GuardDecision(allowed: bool, state: str, reason: str, adjusted_quote_qty: float | None)`
- `OrderIntent(symbol, side, quote_qty, quantity, confidence, data_age_seconds)`
- `AccountRiskSnapshot(equity_usdt, daily_pnl_fraction, consecutive_losses)`

Implement the minimal deterministic checks. Do not call Binance from this module.

**Step 3: Verify**

Run:
`PYTHONPATH=src uv run python -m unittest tests.test_live_guard`

Expected: PASS.

---

### Task 3: Safe Order Gateway Around Broker

**Files:**
- Create: `src/bat/execution/order_gateway.py`
- Modify: `src/bat/tui/app.py:834-905`
- Test: `tests/test_order_gateway.py`

**Step 1: Write failing tests**

Use a fake broker object with `buy()` and `sell()` counters.

Test:
- blocked guard decision never calls broker
- `SHADOW` records intended order but never calls broker
- `CANARY` BUY calls broker with adjusted `quote_qty`
- `CANARY` SELL calls broker with adjusted `quantity`
- gateway returns a structured result with `sent`, `blocked_reason`, and `order_id`

Run:
`PYTHONPATH=src uv run python -m unittest tests.test_order_gateway`

Expected: FAIL.

**Step 2: Implement gateway**

Add `OrderGateway.submit(intent, guard_context)` that:
- calls `LiveGuard`
- logs/returns blocked decisions
- only calls `BinanceBroker` when guard allows
- keeps `BinanceBroker` unchanged for now

**Step 3: Wire TUI through gateway**

Replace direct `broker.buy()` / `broker.sell()` calls in `action_auto_trade()` with `OrderGateway`.

Do not enable REAL by default.

**Step 4: Verify**

Run:
`PYTHONPATH=src uv run python -m unittest tests.test_order_gateway`

Expected: PASS.

---

### Task 4: Data Quality Gate For Live Decisions

**Files:**
- Create: `src/bat/data/live_quality.py`
- Modify: `src/bat/tui/app.py:851-864`
- Test: `tests/test_live_data_quality.py`

**Step 1: Write failing tests**

Test quality gate detects:
- too few klines
- non-monotonic timestamps
- stale latest candle
- missing candle gap
- zero or negative OHLCV values
- extreme one-candle return beyond configured threshold

Run:
`PYTHONPATH=src uv run python -m unittest tests.test_live_data_quality`

Expected: FAIL.

**Step 2: Implement pure quality checker**

Add:
- `LiveDataQualityResult(ok: bool, reason: str, latest_close_time_ms: int | None, data_age_seconds: float | None)`
- `check_live_klines(klines, interval="15m", max_age_seconds=1200)`

No Binance calls. Pure function only.

**Step 3: Wire into TUI loop**

Before `AnalystAgent.analyze(klines)`, call quality checker. If not OK, log reason and skip trade decision.

**Step 4: Verify**

Run:
`PYTHONPATH=src uv run python -m unittest tests.test_live_data_quality`

Expected: PASS.

---

### Task 5: Model Admission Gate And Walk-Forward Requirement

**Files:**
- Create: `src/bat/services/model_admission.py`
- Modify: `src/bat/training.py`
- Test: `tests/test_model_admission.py`

**Step 1: Write failing tests**

Test admission rejects runs when:
- no `report.json`
- `expected_return_after_cost <= 0`
- `profit_factor < 1.2`
- `max_drawdown < -0.10`
- `trade_count < 30`
- `model_family` missing
- validation method is only one holdout and no walk-forward summary exists

Test admission accepts a HistGB run with:
- positive expected return
- profit factor above threshold
- max drawdown above threshold
- sufficient trade count
- required artifacts present

Run:
`PYTHONPATH=src uv run python -m unittest tests.test_model_admission`

Expected: FAIL.

**Step 2: Implement admission report**

Add:
- `ModelAdmissionResult(accepted: bool, reasons: list[str], metrics: dict)`
- `evaluate_run_for_live(run_path, require_walk_forward=True)`

Keep thresholds configurable but safe by default.

**Step 3: Add temporary holdout warning**

Until full walk-forward is implemented, single-holdout HistGB can be used for PAPER/SHADOW only, not CANARY.

**Step 4: Verify**

Run:
`PYTHONPATH=src uv run python -m unittest tests.test_model_admission`

Expected: PASS.

---

### Task 6: Explainability Artifacts For HistGB

**Files:**
- Create: `src/bat/services/explainability.py`
- Modify: `src/bat/training.py`
- Test: `tests/test_explainability_artifacts.py`

**Step 1: Write failing tests**

Test a HistGB run writes:
- `feature_importance.json`
- `decision_thresholds.json`
- `explainability_report.json`

Because `HistGradientBoostingClassifier` does not expose classic impurity importances, use permutation importance on validation data or a deterministic fallback report if sample size is too small.

Run:
`PYTHONPATH=src uv run python -m unittest tests.test_explainability_artifacts`

Expected: FAIL.

**Step 2: Implement minimal explainability service**

Add:
- `write_histgb_explainability(run, model, x_valid, y_valid, feature_columns)`
- top feature list
- validation sample count
- thresholds used for confidence/edge filters

**Step 3: Verify**

Run:
`PYTHONPATH=src uv run python -m unittest tests.test_explainability_artifacts`

Expected: PASS.

---

### Task 7: TUI Live Readiness Panel

**Files:**
- Modify: `src/bat/tui/app.py`
- Test: `tests/test_tui_live_readiness.py`

**Step 1: Write failing tests**

Source-level tests should verify the TUI contains:
- live state selector
- canary risk display
- model admission status display
- guard block reason display
- default state text containing PAPER or OFF

Run:
`PYTHONPATH=src uv run python -m unittest tests.test_tui_live_readiness`

Expected: FAIL.

**Step 2: Add UI fields**

Add controls/status under the trading tab or training tab:
- `select_live_state`
- `live_readiness_status`
- `guard_status`
- `model_admission_status`

Do not make REAL/CANARY easy to toggle accidentally; state changes must validate with `validate_transition()`.

**Step 3: Verify**

Run:
`PYTHONPATH=src uv run python -m unittest tests.test_tui_live_readiness`

Expected: PASS.

---

### Task 8: Operational Runbook And Verification

**Files:**
- Create: `docs/live-readiness-runbook.md`
- Create: `docs/model-admission.md`
- Modify: `README.md` if present

**Step 1: Write runbook**

Document:
- how to run PAPER
- how to run SHADOW
- CANARY prerequisites
- kill switch procedure
- what metrics block live trading
- how to inspect guard/model/data failures

**Step 2: Full verification**

Run:
`PYTHONPATH=src uv run python -m unittest discover -s tests`

Expected: all tests pass.

Run:
`PYTHONPATH=src uv run python -m compileall src tests`

Expected: no compile errors.

**Step 3: Manual smoke**

Run PAPER/SHADOW only with `BTCUSDT 15m` and verify:
- blocked real order reasons are logged
- no real order is submitted in PAPER/SHADOW
- guard status appears in TUI logs

**Step 4: Git checkpoint**

Inspect:
- `git status --short`
- `git diff`

Do not commit unless the user explicitly requests it.

---

## Later Phases Not In This Plan

- Multi-symbol generalization beyond BTCUSDT.
- Portfolio-level risk allocation.
- Model ensemble and regime detection.
- Full live order reconciliation dashboard.
- Real CANARY execution. This plan builds the safety gates required before enabling it.
