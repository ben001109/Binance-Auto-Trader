from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path


@dataclass(frozen=True)
class AdmissionCriteria:
    min_expected_return_after_cost: float = 0.0
    min_profit_factor: float = 1.0
    max_drawdown_floor: float = -0.2
    min_trade_count: int = 30
    max_prediction_fraction: float = 0.9
    min_non_hold_fraction: float = 0.02
    min_fold_count: int = 2


@dataclass(frozen=True)
class AdmissionDecision:
    passed: bool
    reasons: list[str]
    criteria: AdmissionCriteria

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "reasons": list(self.reasons),
            "criteria": asdict(self.criteria),
        }


def assess_model_admission(
    metrics: dict,
    artifact_paths: list[str | Path] | tuple[str | Path, ...] | None = None,
    criteria: AdmissionCriteria | None = None,
) -> AdmissionDecision:
    criteria = criteria or AdmissionCriteria()
    reasons: list[str] = []

    if int(metrics.get("fold_count", 1) or 0) < criteria.min_fold_count:
        reasons.append("fold_count below minimum")
    expected_return = _finite_metric(metrics, "expected_return_after_cost", reasons)
    profit_factor = _finite_metric(metrics, "profit_factor", reasons)
    max_drawdown = _finite_metric(metrics, "max_drawdown", reasons)

    if expected_return < criteria.min_expected_return_after_cost:
        reasons.append("expected_return_after_cost below minimum")
    if profit_factor < criteria.min_profit_factor:
        reasons.append("profit_factor below minimum")
    if max_drawdown < criteria.max_drawdown_floor:
        reasons.append("max_drawdown below floor")
    if int(metrics.get("trade_count", 0) or 0) < criteria.min_trade_count:
        reasons.append("trade_count below minimum")

    distribution = _normalize_distribution(metrics.get("prediction_distribution", {}))
    total_predictions = sum(distribution.values())
    if total_predictions <= 0:
        reasons.append("prediction distribution unavailable")
    else:
        max_fraction = max(distribution.values()) / total_predictions
        non_hold_fraction = (distribution.get(0, 0) + distribution.get(2, 0)) / total_predictions
        if max_fraction > criteria.max_prediction_fraction:
            reasons.append("prediction distribution too concentrated")
        if non_hold_fraction < criteria.min_non_hold_fraction:
            reasons.append("non-hold prediction fraction below minimum")

    for path in artifact_paths or []:
        if not Path(path).exists():
            reasons.append("required artifact missing")
            break

    return AdmissionDecision(passed=not reasons, reasons=reasons, criteria=criteria)


def _normalize_distribution(value) -> dict[int, int]:
    if not isinstance(value, dict):
        return {}
    distribution: dict[int, int] = {}
    for key, count in value.items():
        try:
            distribution[int(key)] = int(count)
        except (TypeError, ValueError):
            continue
    return {label: distribution.get(label, 0) for label in (0, 1, 2)}


def _finite_metric(metrics: dict, key: str, reasons: list[str]) -> float:
    try:
        value = float(metrics.get(key, 0.0))
    except (TypeError, ValueError):
        value = math.nan
    if not math.isfinite(value):
        reasons.append(f"{key} unavailable")
        return 0.0
    return value
