from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import pickle
import re
import subprocess

import torch
import yaml

from bat.research_config import ResearchConfig
from bat.services.artifact_security import sha256_file, write_artifact_manifest


TRAINING_LOG_HEADER = [
    "epoch",
    "train_loss",
    "val_loss",
    "val_accuracy",
    "val_macro_f1",
    "expected_return_after_cost",
    "win_rate",
    "profit_factor",
    "max_drawdown",
]


@dataclass(frozen=True)
class TrainingRun:
    path: Path
    feature_columns: list[str] | None = None
    config_sha256: str | None = None
    training_data_sha256: str | None = None
    git_sha: str | None = None

    @property
    def best_model_path(self) -> Path:
        return self.path / "best_model.pt"

    @property
    def best_sklearn_model_path(self) -> Path:
        return self.path / "best_model.pkl"

    def save_best_model(self, model) -> None:
        torch.save(model.state_dict(), self.best_model_path)
        self._write_manifest_artifact(
            self.best_model_path,
            artifact_type="torch_state_dict",
            runtime_load_allowed=True,
        )

    def save_sklearn_model(self, model) -> None:
        self.best_sklearn_model_path.write_bytes(pickle.dumps(model))
        self._write_manifest_artifact(
            self.best_sklearn_model_path,
            artifact_type="pickle",
            runtime_load_allowed=False,
        )

    def _write_manifest_artifact(
        self,
        artifact_path: Path,
        *,
        artifact_type: str,
        runtime_load_allowed: bool,
    ) -> None:
        write_artifact_manifest(
            self.path,
            [
                {
                    "path": artifact_path,
                    "type": artifact_type,
                    "runtime_load_allowed": runtime_load_allowed,
                }
            ],
            feature_columns=list(self.feature_columns or []),
            config_sha256=self.config_sha256,
            training_data_sha256=self.training_data_sha256,
            git_sha=self.git_sha,
        )

    def write_epoch_metrics(self, metrics: dict) -> None:
        row = [metrics.get(column, "") for column in TRAINING_LOG_HEADER]
        with (self.path / "training_log.csv").open("a", encoding="utf-8") as handle:
            handle.write(",".join(str(value) for value in row) + "\n")
        (self.path / "metrics.json").write_text(
            json.dumps({"latest": metrics}, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    def write_report(self, report: dict) -> None:
        (self.path / "report.json").write_text(
            json.dumps(report, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )


class RunManager:
    def __init__(self, root: str | Path = "runs"):
        self.root = Path(root)

    def create_run(
        self,
        config: ResearchConfig,
        model_name: str,
        feature_columns: list[str],
        label_distribution: dict[int, int],
        timestamp: str | None = None,
    ) -> TrainingRun:
        stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        symbol = self._safe_name(config.data.symbol)
        interval = self._safe_name(config.data.interval)
        model = self._safe_name(model_name)
        run_path = self.root / f"{stamp}_{model}_{symbol}_{interval}"
        run_path.mkdir(parents=True, exist_ok=False)

        config_path = run_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump(asdict(config), sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
        (run_path / "training_log.csv").write_text(
            ",".join(TRAINING_LOG_HEADER) + "\n",
            encoding="utf-8",
        )
        self._write_json(run_path / "metrics.json", {})
        self._write_json(run_path / "report.json", {"status": "initialized"})
        self._write_json(run_path / "feature_columns.json", list(feature_columns))
        self._write_json(
            run_path / "label_distribution.json",
            {str(key): int(value) for key, value in label_distribution.items()},
        )
        config_sha256 = sha256_file(config_path)
        write_artifact_manifest(
            run_path,
            [],
            feature_columns=list(feature_columns),
            config_sha256=config_sha256,
            training_data_sha256=None,
            git_sha=self._git_sha(),
        )
        return TrainingRun(
            path=run_path,
            feature_columns=list(feature_columns),
            config_sha256=config_sha256,
            training_data_sha256=None,
            git_sha=self._git_sha(),
        )

    def _write_json(self, path: Path, payload) -> None:
        path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    def _safe_name(self, value: str) -> str:
        cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
        return cleaned.strip("_") or "unknown"

    def _git_sha(self) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=Path.cwd(),
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=2,
            ).strip()
        except Exception:
            return None
