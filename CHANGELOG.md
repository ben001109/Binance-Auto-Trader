# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **Simulation**: Fixed `AttributeError` caused by unsafe f-string formatting when `risk` is None.
- **Analysis**: Fixed `ValueError` in LSTM strategy by falling back to simple logic when training data is insufficient for multi-timeframe features.

## [1.0.0] - 2026-01-10

### Added
- **CI/CD**:
    - GitHub Actions workflow (`ci.yml`) for automated testing on push/PR.
    - Release workflow (`release.yml`) for multi-platform builds (Windows, macOS, Linux).
    - `bat_build.spec` for robust PyInstaller bundling using `uv`.
- **TUI Enhancements**:
    - `PerformanceMonitor` widget showing CPU, RAM, and GPU/VRAM usage (support for CUDA & Apple MPS).
    - Persistent progress tracking for history download and simulation in the "Training Status" panel.
- **Data Integrity**:
    - New `integrity.py` module for detecting and healing missing kline data.
    - Automatic integrity checks on application startup (`ensure_data_integrity`).
- **Strategy & Model**:
    - Multi-timeframe features: Added 1-hour RSI and EMA indicators.
    - Cost-aware training targets: Dynamic thresholds based on trading fees and slippage.
    - Risk Profile presets (Conservative, Standard, Aggressive) in TUI.
- **Verification**:
    - `verify_enhancements.py` and `verify_analyst.py` for regression and smoke testing.

### Fixed
- **TUI**: Corrected issue where progress bars would scroll off-screen in logs; now pinned to status panel.
- **Data Processing**:
    - Fixed `TypeError` in `add_1h_features` caused by unstable timestamp types during resampling.
    - Fixed `KeyError` by ensuring HTF feature columns are always initialized.
- **Recursion Bugs**: Resolved infinite recursion in verification scripts.

### Changed
- **Roadmap**: Updated `TODO.md` with new features (Notifications, Docker, Trailing Stops).
- **Architecture**: Deepened integration of `AnalystAgent` as the central trading brain.
