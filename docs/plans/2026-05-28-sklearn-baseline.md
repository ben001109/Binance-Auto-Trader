# Sklearn Baseline Integration Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a research baseline model path so `HistGradientBoosting` can be trained and evaluated through the same dataset, validation, decision filter, and run artifact pipeline as CNN-LSTM.

**Architecture:** Keep torch sequence models unchanged. Add a separate sklearn branch in `train_model()` for research models such as `histgb`, using flat row-level features aligned to the sequence target index, then write the same metrics/report artifacts plus `best_model.pkl`.

**Tech Stack:** Python 3.13, scikit-learn, numpy, unittest, uv.

---

### Task 1: Model Name Support

**Files:**
- Modify: `src/bat/models/factory.py`
- Test: `tests/test_model_factory.py`

**Steps:**
1. Write failing tests for `normalize_model_name("hist_gradient_boosting") == "histgb"` and unsupported names still raising.
2. Run `PYTHONPATH=src uv run python -m unittest tests.test_model_factory` and confirm RED.
3. Implement minimal normalization support for `histgb` aliases without changing torch model construction.
4. Run the same targeted tests and confirm GREEN.

### Task 2: Sklearn Artifact Support

**Files:**
- Modify: `src/bat/services/run_manager.py`
- Test: `tests/test_run_manager.py`

**Steps:**
1. Write a failing test that saves a pickle model artifact as `best_model.pkl`.
2. Run `PYTHONPATH=src uv run python -m unittest tests.test_run_manager` and confirm RED.
3. Add `TrainingRun.best_sklearn_model_path` and `TrainingRun.save_sklearn_model()`.
4. Run targeted tests and confirm GREEN.

### Task 3: HistGB Training Branch

**Files:**
- Modify: `src/bat/training.py`
- Test: `tests/test_training_run_artifacts.py`

**Steps:**
1. Write a failing test for `train_model(... research_config=ResearchConfig(model=ModelConfig(name="histgb")))` producing `best_model.pkl`, `training_log.csv`, and `report.json`.
2. Run `PYTHONPATH=src uv run python -m unittest tests.test_training_run_artifacts` and confirm RED.
3. Implement helpers to build flat train/validation arrays from `TrainingInputs` and evaluate sklearn probabilities through the existing decision filters.
4. Route research `histgb` model names to the sklearn branch before torch model setup.
5. Run targeted tests and confirm GREEN.

### Task 4: Verification And Smoke Run

**Commands:**
- `PYTHONPATH=src uv run python -m unittest discover -s tests`
- `PYTHONPATH=src uv run python -m compileall src tests`
- Run one 1-year BTCUSDT histgb baseline with `horizon=16`, `min_edge=0.002`, `confidence=0.45`, `edge=0.08` and inspect run artifacts.

**Scope Guard:**
- Do not replace CNN-LSTM.
- Do not wire sklearn baseline into live trading execution yet.
- Do not change live/testnet trading defaults.
- Do not commit unless explicitly requested.
