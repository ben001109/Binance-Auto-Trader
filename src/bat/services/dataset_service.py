from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bat.data.dataset import ResearchSequenceDataset, build_cost_aware_labels
from bat.data.features import DEFAULT_RESEARCH_FEATURES, DataQualityReport, add_research_features, clean_ohlcv
from bat.data.normalization import rolling_zscore_normalize
from bat.research_config import ResearchConfig


@dataclass(frozen=True)
class DatasetSummary:
    rows: int
    start_time: str | None
    end_time: str | None
    missing_candles: int
    nan_count: int
    feature_count: int
    selected_features: list[str]
    label_distribution: dict[int, int]
    normalization_mode: str
    seq_len: int
    horizon: int
    fee: float
    slippage: float
    min_edge: float
    warnings: list[str]
    cache_status: str = "disabled"


@dataclass(frozen=True)
class PreparedDataset:
    features: np.ndarray
    labels: np.ndarray
    future_returns: np.ndarray
    dataset: ResearchSequenceDataset
    feature_columns: list[str]
    quality_report: DataQualityReport
    summary: DatasetSummary


class DatasetService:
    def __init__(self, config: ResearchConfig):
        self.config = config

    def prepare_csv(self, path: str | Path) -> PreparedDataset:
        df = pd.read_csv(path)
        return self.prepare_dataframe(df)

    def prepare_csv_cached(
        self,
        path: str | Path,
        cache_dir: str | Path = "data/cache/research_dataset",
    ) -> PreparedDataset:
        source_path = Path(path)
        cache_path = Path(cache_dir)
        expected_meta = self._cache_metadata(source_path)
        features_path = cache_path / "features.npy"
        labels_path = cache_path / "labels.npy"
        future_returns_path = cache_path / "future_returns.npy"
        meta_path = cache_path / "meta.json"

        if features_path.exists() and labels_path.exists() and future_returns_path.exists() and meta_path.exists():
            try:
                cached_meta = json.loads(meta_path.read_text(encoding="utf-8"))
                cached_identity = {
                    key: value for key, value in cached_meta.items() if key != "quality_report"
                }
                if cached_identity == expected_meta:
                    features, labels, future_returns = self._load_valid_cached_arrays(
                        features_path,
                        labels_path,
                        future_returns_path,
                    )
                    report = self._quality_report_from_meta(cached_meta)
                    summary = self._build_summary(report, features, labels, cache_status="hit")
                    return PreparedDataset(
                        features=features,
                        labels=labels,
                        future_returns=future_returns,
                        dataset=ResearchSequenceDataset(features, labels, self.config.data.seq_len),
                        feature_columns=list(DEFAULT_RESEARCH_FEATURES),
                        quality_report=report,
                        summary=summary,
                    )
            except Exception:
                pass

        prepared = self.prepare_csv(source_path)
        cache_path.mkdir(parents=True, exist_ok=True)
        metadata = dict(expected_meta)
        metadata["quality_report"] = self._quality_report_to_dict(prepared.quality_report)
        np.save(features_path, prepared.features)
        np.save(labels_path, prepared.labels)
        np.save(future_returns_path, prepared.future_returns)
        meta_path.write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")
        summary = self._build_summary(
            prepared.quality_report,
            prepared.features,
            prepared.labels,
            cache_status="miss",
        )
        return PreparedDataset(
            features=prepared.features,
            labels=prepared.labels,
            future_returns=prepared.future_returns,
            dataset=prepared.dataset,
            feature_columns=prepared.feature_columns,
            quality_report=prepared.quality_report,
            summary=summary,
        )

    def prepare_dataframe(self, df: pd.DataFrame) -> PreparedDataset:
        cleaned, report = clean_ohlcv(df, expected_interval=self.config.data.interval)
        featured = add_research_features(cleaned)
        labels = build_cost_aware_labels(
            featured,
            horizon=self.config.data.horizon,
            fee=self.config.label.fee,
            slippage=self.config.label.slippage,
            min_edge=self.config.label.min_edge,
        )
        future_returns = featured["close"].shift(-self.config.data.horizon) / featured["close"] - 1
        normalized = rolling_zscore_normalize(
            featured,
            DEFAULT_RESEARCH_FEATURES,
            window=self.config.data.rolling_norm_window,
        )
        usable = normalized[DEFAULT_RESEARCH_FEATURES].copy()
        usable["label"] = labels
        usable["future_return"] = future_returns
        usable = usable.replace([np.inf, -np.inf], np.nan).dropna()

        features = usable[DEFAULT_RESEARCH_FEATURES].to_numpy(dtype=np.float32)
        label_values = usable["label"].astype(int).to_numpy(dtype=np.int64)
        future_return_values = usable["future_return"].to_numpy(dtype=float)
        dataset = ResearchSequenceDataset(features, label_values, self.config.data.seq_len)
        summary = self._build_summary(report, features, label_values)
        return PreparedDataset(
            features=features,
            labels=label_values,
            future_returns=future_return_values,
            dataset=dataset,
            feature_columns=list(DEFAULT_RESEARCH_FEATURES),
            quality_report=report,
            summary=summary,
        )

    def _build_summary(
        self,
        report: DataQualityReport,
        features: np.ndarray,
        labels: np.ndarray,
        cache_status: str = "disabled",
    ) -> DatasetSummary:
        distribution = {label: int((labels == label).sum()) for label in (0, 1, 2)}
        warnings = []
        total = max(len(labels), 1)
        if distribution.get(1, 0) / total > 0.85:
            warnings.append("Label imbalance too high")
        if len(features) <= self.config.data.seq_len:
            warnings.append("Insufficient rows for sequence dataset")

        return DatasetSummary(
            rows=int(len(features)),
            start_time=report.start_time,
            end_time=report.end_time,
            missing_candles=report.missing_candles,
            nan_count=report.nan_count,
            feature_count=len(DEFAULT_RESEARCH_FEATURES),
            selected_features=list(DEFAULT_RESEARCH_FEATURES),
            label_distribution=distribution,
            normalization_mode="rolling_zscore",
            seq_len=self.config.data.seq_len,
            horizon=self.config.data.horizon,
            fee=self.config.label.fee,
            slippage=self.config.label.slippage,
            min_edge=self.config.label.min_edge,
            warnings=warnings,
            cache_status=cache_status,
        )

    def _cache_metadata(self, source_path: Path) -> dict:
        stat = source_path.stat()
        history_meta_path = source_path.with_suffix(".meta.json")
        history_meta = {}
        if history_meta_path.exists():
            try:
                loaded = json.loads(history_meta_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    history_meta = loaded
            except Exception:
                history_meta = {}
        return {
            "version": 3,
            "source_path": str(source_path.resolve()),
            "source_sha256": self._file_sha256(source_path),
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "history_meta": history_meta,
            "feature_columns": list(DEFAULT_RESEARCH_FEATURES),
            "config_fingerprint": self._config_fingerprint(),
        }

    def _config_fingerprint(self) -> str:
        payload = {
            "data": {
                "seq_len": self.config.data.seq_len,
                "horizon": self.config.data.horizon,
                "rolling_norm_window": self.config.data.rolling_norm_window,
                "interval": self.config.data.interval,
            },
            "label": {
                "fee": self.config.label.fee,
                "slippage": self.config.label.slippage,
                "min_edge": self.config.label.min_edge,
            },
            "feature_columns": list(DEFAULT_RESEARCH_FEATURES),
        }
        encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _file_sha256(self, path: str | Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _load_valid_cached_arrays(
        self,
        features_path: Path,
        labels_path: Path,
        future_returns_path: Path,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        features = np.load(features_path, allow_pickle=False)
        labels = np.load(labels_path, allow_pickle=False)
        future_returns = np.load(future_returns_path, allow_pickle=False)
        self._validate_cached_arrays(features, labels, future_returns)
        return features, labels, future_returns

    def _validate_cached_arrays(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        future_returns: np.ndarray,
    ) -> None:
        if features.ndim != 2:
            raise ValueError("Cached features must be a 2D array")
        if features.shape[1] != len(DEFAULT_RESEARCH_FEATURES):
            raise ValueError("Cached feature width does not match feature columns")
        if labels.ndim != 1 or future_returns.ndim != 1:
            raise ValueError("Cached labels and returns must be 1D arrays")
        if len(features) != len(labels) or len(labels) != len(future_returns):
            raise ValueError("Cached array lengths do not match")
        if features.dtype != np.float32:
            raise ValueError("Cached features must be float32")
        if not np.issubdtype(labels.dtype, np.integer):
            raise ValueError("Cached labels must be integer dtype")
        if not np.issubdtype(future_returns.dtype, np.number):
            raise ValueError("Cached future returns must be numeric")
        if not np.isfinite(features).all() or not np.isfinite(future_returns).all():
            raise ValueError("Cached arrays must be finite")
        unique_labels = set(labels.astype(int).tolist())
        if not unique_labels.issubset({0, 1, 2}):
            raise ValueError("Cached labels out of range")

    def _quality_report_to_dict(self, report: DataQualityReport) -> dict:
        return {
            "rows_before": report.rows_before,
            "rows_after": report.rows_after,
            "duplicate_timestamps": report.duplicate_timestamps,
            "dropped_nan_rows": report.dropped_nan_rows,
            "missing_candles": report.missing_candles,
            "nan_count": report.nan_count,
            "start_time": report.start_time,
            "end_time": report.end_time,
        }

    def _quality_report_from_meta(self, metadata: dict) -> DataQualityReport:
        report = metadata.get("quality_report", {})
        return DataQualityReport(
            rows_before=int(report.get("rows_before", 0)),
            rows_after=int(report.get("rows_after", 0)),
            duplicate_timestamps=int(report.get("duplicate_timestamps", 0)),
            dropped_nan_rows=int(report.get("dropped_nan_rows", 0)),
            missing_candles=int(report.get("missing_candles", 0)),
            nan_count=int(report.get("nan_count", 0)),
            start_time=report.get("start_time"),
            end_time=report.get("end_time"),
        )
