# Research Data Pipeline Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build the first safe MVP of BAT's research data pipeline: YAML config, leakage-safe feature normalization, cost-aware labels, and sequence datasets.

**Architecture:** Add a parallel research pipeline next to the existing bot flow. Existing `Config`, TUI, LSTM training, and backtest entry points stay functional while new modules become testable building blocks for later trainer/TUI integration.

**Tech Stack:** Python 3.13, pandas, numpy, torch, pandas-ta, PyYAML, unittest, uv.

---

### Task 1: YAML Config Loader

**Files:**
- Create: `configs/training.yaml`
- Create: `src/bat/research_config.py`
- Modify: `pyproject.toml`
- Test: `tests/test_research_config.py`

**Steps:**
1. Write failing tests that load YAML into dataclasses and verify defaults for paper-only risk settings.
2. Add PyYAML dependency.
3. Implement typed dataclasses and `load_research_config()`.
4. Run `PYTHONPATH=src uv run python -m unittest tests.test_research_config`.

### Task 2: Leakage-Safe Data Pipeline

**Files:**
- Create: `src/bat/data/features.py`
- Create: `src/bat/data/normalization.py`
- Modify: `src/bat/data/dataset.py`
- Test: `tests/test_data_pipeline.py`

**Steps:**
1. Write failing tests for OHLCV cleaning report, feature columns, rolling z-score shift behavior, cost-aware labels, and sequence shapes.
2. Implement minimal data cleaning and feature engineering.
3. Implement rolling normalization using shifted rolling mean/std.
4. Implement label builder and lazy `SequenceDataset`.
5. Run `PYTHONPATH=src uv run python -m unittest tests.test_data_pipeline`.

### Task 3: Time-Ordered Split Foundation

**Files:**
- Create: `src/bat/data/split.py`
- Test: `tests/test_data_split.py`

**Steps:**
1. Write failing tests for walk-forward fold order and non-overlap.
2. Implement duration parsing and walk-forward fold generation.
3. Run `PYTHONPATH=src uv run python -m unittest tests.test_data_split`.

### Task 4: Verification

**Commands:**
- `PYTHONPATH=src uv run python -m unittest discover -s tests`
- `PYTHONPATH=src uv run python -m compileall src tests`

**Scope Guard:**
- Do not replace `src/bat/training.py` in this phase.
- Do not alter live trading defaults.
- Do not add CNN-LSTM or trainer service yet.
- Keep TUI integration for a follow-up patch after the data pipeline is verified.
