# Early Stopping Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Stop training when the configured validation metric stops improving, save the best model, and record the stop reason in run artifacts and TUI logs.

**Architecture:** Add a small pure helper in `src/bat/training.py` to track best metric, best epoch, patience, and stop reason. Integrate it after validation metrics are written each epoch, leaving the existing training loop and legacy pipeline behavior intact.

**Tech Stack:** Python 3.13, torch, unittest, uv.

---

### Task 1: Early Stopping State Helper

**Files:**
- Modify: `src/bat/training.py`
- Test: `tests/test_early_stopping.py`

**Steps:**
1. Write failing tests for first metric improvement, patience exhaustion, missing metric handling, and lower-is-better `val_loss` behavior.
2. Run `PYTHONPATH=src uv run python -m unittest tests.test_early_stopping` and confirm the tests fail because `_update_early_stopping` does not exist.
3. Implement a minimal `EarlyStoppingState` dataclass and `_update_early_stopping()` helper.
4. Run `PYTHONPATH=src uv run python -m unittest tests.test_early_stopping` and confirm it passes.

### Task 2: Trainer Integration

**Files:**
- Modify: `src/bat/training.py`
- Test: `tests/test_training_run_artifacts.py`

**Steps:**
1. Write a failing training test that drives flat validation metrics, expects training to stop before all requested epochs, expects best-model saving on improvement only, and expects `report.json` to contain the early stop reason.
2. Run `PYTHONPATH=src uv run python -m unittest tests.test_training_run_artifacts` and confirm the new test fails because trainer integration is missing.
3. Integrate `_update_early_stopping()` after validation evaluation in `train_model()`.
4. Save the run best model when the monitored metric improves.
5. Write `report.json` with status, best epoch, best metric, monitored metric, patience, completed epochs, and stop reason.
6. Send `on_log` and `on_status` early stop details when patience is exhausted.
7. Run `PYTHONPATH=src uv run python -m unittest tests.test_training_run_artifacts tests.test_early_stopping` and confirm both pass.

### Task 3: Verification

**Commands:**
- `PYTHONPATH=src uv run python -m unittest discover -s tests`
- `PYTHONPATH=src uv run python -m compileall src tests`

**Scope Guard:**
- Do not change live-trading defaults.
- Do not replace the trainer service yet.
- Keep legacy model training functional.
