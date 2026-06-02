from __future__ import annotations

import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RiskStateSnapshot:
    daily_pnl_fraction: float
    consecutive_losses: int
    available: bool


def latest_kline_data_age_seconds(klines: Any, now_ms: int | float | None = None) -> float:
    try:
        if not klines:
            return math.nan
        close_time_ms = float(klines[-1][6])
        current_ms = float(now_ms if now_ms is not None else time.time() * 1000.0)
        age_seconds = (current_ms - close_time_ms) / 1000.0
    except (TypeError, ValueError, IndexError):
        return math.nan
    if not math.isfinite(close_time_ms) or not math.isfinite(age_seconds):
        return math.nan
    if age_seconds < 0:
        return math.nan
    return age_seconds


def risk_state_from_trade_ledger(path: str | Path, *, is_testnet: bool) -> RiskStateSnapshot:
    ledger_path = Path(path)
    if not ledger_path.exists():
        return _default_snapshot(is_testnet)
    try:
        with ledger_path.open("r", encoding="utf-8", newline="") as handle:
            latest = None
            for latest in csv.DictReader(handle):
                pass
    except (OSError, csv.Error, UnicodeDecodeError):
        return _default_snapshot(is_testnet)

    if latest is None:
        return _default_snapshot(is_testnet)

    try:
        daily_pnl_fraction = float(latest["daily_pnl_fraction"])
        consecutive_losses = int(latest["consecutive_losses"])
    except (KeyError, TypeError, ValueError):
        return _default_snapshot(is_testnet)

    if not math.isfinite(daily_pnl_fraction) or consecutive_losses < 0:
        return _default_snapshot(is_testnet)
    return RiskStateSnapshot(daily_pnl_fraction, consecutive_losses, True)


def _default_snapshot(is_testnet: bool) -> RiskStateSnapshot:
    if is_testnet:
        return RiskStateSnapshot(0.0, 0, True)
    return RiskStateSnapshot(math.nan, 0, False)
