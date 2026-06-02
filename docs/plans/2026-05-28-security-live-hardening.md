# Security Live Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Close the security and trading-safety findings found in the project audit before any real BTCUSDT trading is allowed.

**Architecture:** Add defense-in-depth around every order path, then make live guard inputs authoritative instead of placeholders. Harden logs, model artifacts, data cache, and validation so a model cannot be promoted or used live without provenance, freshness, and admission evidence.

**Tech Stack:** Python 3.13, unittest, Textual TUI, Binance Spot SDK, PyTorch, scikit-learn, uv.

---

## Security Audit Rule

Every task below must finish with:
- targeted tests for the changed area
- `PYTHONPATH=src uv run python -m unittest discover -s tests`
- `PYTHONPATH=src uv run python -m compileall src tests`
- a scoped security review of the changed files before moving to the next task

---

### Task 1: Real Order Interlock And Broker Defense-In-Depth

**Files:**
- Modify: `src/bat/execution/broker.py`
- Modify: `scripts/demo_trade.py`
- Test: `tests/test_broker_live_interlock.py`

**Goal:** No direct broker path can place real orders unless an explicit environment interlock is present.

**Steps:**
1. Write tests that `BinanceBroker(is_testnet=False).buy()` and `.sell()` refuse real orders when `BAT_ALLOW_REAL_TRADING` is not exactly `I_UNDERSTAND_REAL_RISK`.
2. Write tests that testnet orders are not blocked by the interlock.
3. Implement `BinanceBroker._real_trading_interlock_allows_orders()` and check it at the start of `buy()` / `sell()`.
4. Update `scripts/demo_trade.py` to force `is_testnet=True` or fail closed in REAL mode.
5. Run targeted tests and security review.

### Task 2: Route Manual TUI Sell Through Safety Gateway

**Files:**
- Modify: `src/bat/tui/app.py`
- Test: `tests/test_manual_sell_safety.py`

**Goal:** Manual sell cannot bypass `OrderGateway` / `LiveGuard` in REAL mode.

**Steps:**
1. Write tests or source-level assertions that `action_sell_asset()` no longer directly calls `broker.sell()`.
2. Add gateway submission for manual sell with `OrderIntent(side="SELL")`.
3. In TESTNET, keep behavior safe; in REAL, default config must block unless CANARY and interlock pass.
4. Log structured blocked reasons.
5. Run tests and security review.

### Task 3: Live Risk Snapshot And Data Freshness Must Be Real Or Fail Closed

**Files:**
- Create: `src/bat/risk/risk_state.py`
- Modify: `src/bat/tui/app.py`
- Test: `tests/test_live_risk_state.py`

**Goal:** Auto-trade must not pass dummy `daily_pnl_fraction=0.0`, `consecutive_losses=0`, or `data_age_seconds=0.0` in REAL/CANARY mode.

**Steps:**
1. Write tests for calculating kline data age from latest close time.
2. Write tests that missing/invalid risk state returns NaN or unavailable status, causing `LiveGuard` to block.
3. Implement `RiskStateSnapshot` helpers for daily PnL/consecutive losses from structured ledgers when available.
4. Wire TUI to use computed data age and risk state.
5. Fail closed for REAL when risk state cannot be computed.
6. Run tests and security review.

### Task 4: Idempotent Order Submission And Unknown-State Reconciliation Hook

**Files:**
- Modify: `src/bat/execution/spot_client.py`
- Modify: `src/bat/execution/broker.py`
- Modify: `src/bat/execution/order_gateway.py`
- Test: `tests/test_order_idempotency.py`

**Goal:** A timeout or lost response cannot cause repeated market orders without reconciliation.

**Steps:**
1. Add tests that gateway creates deterministic `client_order_id` for each intent.
2. Pass `newClientOrderId` / SDK equivalent into Binance order payload.
3. Track in-flight order ids per symbol/side in gateway.
4. If broker returns unknown/None, return `execution_unknown` and block further orders until cleared.
5. Run tests and security review.

### Task 5: Structured Real/Paper Audit Ledger And Redacted Logging

**Files:**
- Create: `src/bat/security/redaction.py`
- Create: `src/bat/services/audit_ledger.py`
- Modify: `src/bat/logger.py`
- Modify: `src/bat/execution/order_gateway.py`
- Modify: `src/bat/tui/app.py`
- Test: `tests/test_log_redaction.py`
- Test: `tests/test_audit_ledger.py`

**Goal:** Every allow/block/fail order event is recorded, and logs do not expose secrets or full sensitive financial details by default.

**Steps:**
1. Write tests for redacting API keys, signatures, headers, query params, and order intents.
2. Add logging filter to all BAT file handlers.
3. Write tests for append-only audit ledger records with hash chaining.
4. Record gateway decisions and manual/auto order intents/outcomes.
5. Separate paper/testnet and real ledgers.
6. Run tests and security review.

### Task 6: Safe Model Artifact Loading And Provenance Manifests

**Files:**
- Create: `src/bat/services/artifact_security.py`
- Modify: `src/bat/analyzer.py`
- Modify: `src/bat/training.py`
- Modify: `src/bat/services/run_manager.py`
- Test: `tests/test_artifact_security.py`

**Goal:** Runtime model loading refuses unsafe or unprovenanced artifacts.

**Steps:**
1. Write tests that artifact loader refuses symlinks, paths outside trusted directories, missing manifests, and hash mismatches.
2. Use `torch.load(..., weights_only=True)` where supported.
3. Add `manifest.json` with artifact hashes, config hash, feature columns, training data hash placeholder, git SHA if available.
4. Mark pickle sklearn artifacts as unsafe for runtime unless an explicit trusted loader is implemented.
5. Runtime `analyzer.py` must load only admitted/manifested artifacts.
6. Run tests and security review.

### Task 7: Dataset Cache And OHLCV Hard Validation

**Files:**
- Modify: `src/bat/services/dataset_service.py`
- Modify: `src/bat/data/features.py`
- Test: `tests/test_dataset_cache_security.py`
- Test: `tests/test_ohlcv_quality_gate.py`

**Goal:** Training data and cached arrays cannot be silently poisoned or stale.

**Steps:**
1. Add tests for source CSV hash in cache metadata.
2. Add tests for cached array shape/dtype/finiteness/label range validation.
3. Add OHLCV invariant tests: positive prices, `high >= low`, open/close inside range, non-negative volume, outlier bounds.
4. Implement hash and validation checks.
5. Fail closed on suspicious data.
6. Run tests and security review.

### Task 8: Purged Walk-Forward And Model Admission Gate

**Files:**
- Modify: `src/bat/data/split.py`
- Create: `src/bat/services/model_admission.py`
- Modify: `src/bat/training.py`
- Test: `tests/test_purged_split.py`
- Test: `tests/test_model_admission.py`

**Goal:** No model can be promoted based on leakage-prone single holdout or sparse lucky trades.

**Steps:**
1. Add tests for purge/embargo between train/validation by `horizon`.
2. Honor `validation.method=walk_forward`.
3. Aggregate fold metrics.
4. Implement admission thresholds for expected return, profit factor, max drawdown, trade count, prediction distribution, and artifact presence.
5. Promotion to live model must require admission pass.
6. Run tests and security review.

### Task 9: Real Trading Arming UX And Settings Safety

**Files:**
- Modify: `src/bat/tui/app.py`
- Test: `tests/test_tui_live_readiness.py`

**Goal:** REAL mode cannot be persisted or activated accidentally.

**Steps:**
1. Add tests that `_save_settings()` never persists `REAL` as default startup mode.
2. Add typed arming confirmation state and visible TUI warning.
3. Disable real order buttons unless armed, CANARY-ready, and env interlock is present.
4. Display guard/model/data failure reasons prominently.
5. Run tests and security review.

---

## Completion Gate

Before real CANARY is considered:
- all tasks above pass
- full test suite passes
- compile passes
- final security audit finds no Critical or High findings in order execution, artifact loading, data validation, or TUI arming
- manual PAPER/SHADOW smoke shows no broker submission in blocked states
