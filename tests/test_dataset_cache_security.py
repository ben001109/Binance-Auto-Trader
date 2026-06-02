import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from bat.research_config import DataConfig, LabelConfig, ResearchConfig, TrainingConfig
from bat.services.dataset_service import DatasetService


def sample_history(rows: int = 120) -> pd.DataFrame:
    close = np.linspace(100.0, 130.0, rows)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="15min", tz="UTC"),
            "open": close - 0.2,
            "high": close + 0.8,
            "low": close - 0.8,
            "close": close,
            "volume": np.linspace(10.0, 25.0, rows),
        }
    )


class DatasetCacheSecurityTest(unittest.TestCase):
    def setUp(self):
        self.config = ResearchConfig(
            data=DataConfig(seq_len=8, horizon=2, rolling_norm_window=12),
            label=LabelConfig(fee=0.001, slippage=0.0005, min_edge=0.001),
            training=TrainingConfig(use_preprocessing_cache=True),
        )

    def test_cache_metadata_includes_source_sha256(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "history.csv"
            cache_dir = Path(tmpdir) / "cache"
            sample_history().to_csv(history_path, index=False)

            DatasetService(self.config).prepare_csv_cached(history_path, cache_dir=cache_dir)

            meta = json.loads((cache_dir / "meta.json").read_text(encoding="utf-8"))
            expected = DatasetService(self.config)._file_sha256(history_path)
            self.assertEqual(meta["source_sha256"], expected)
            self.assertEqual(len(meta["source_sha256"]), 64)

    def test_corrupt_cached_features_are_recomputed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "history.csv"
            cache_dir = Path(tmpdir) / "cache"
            sample_history().to_csv(history_path, index=False)
            service = DatasetService(self.config)
            first = service.prepare_csv_cached(history_path, cache_dir=cache_dir)
            poisoned = first.features.copy()
            poisoned[0, 0] = np.nan
            np.save(cache_dir / "features.npy", poisoned)

            second = service.prepare_csv_cached(history_path, cache_dir=cache_dir)

            self.assertEqual(second.summary.cache_status, "miss")
            self.assertTrue(np.isfinite(second.features).all())

    def test_cached_shape_mismatch_is_recomputed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "history.csv"
            cache_dir = Path(tmpdir) / "cache"
            sample_history().to_csv(history_path, index=False)
            service = DatasetService(self.config)
            first = service.prepare_csv_cached(history_path, cache_dir=cache_dir)
            np.save(cache_dir / "labels.npy", first.labels[:-1])

            second = service.prepare_csv_cached(history_path, cache_dir=cache_dir)

            self.assertEqual(second.summary.cache_status, "miss")
            self.assertEqual(len(second.features), len(second.labels))
            self.assertEqual(len(second.labels), len(second.future_returns))

    def test_cached_feature_width_mismatch_is_recomputed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "history.csv"
            cache_dir = Path(tmpdir) / "cache"
            sample_history().to_csv(history_path, index=False)
            service = DatasetService(self.config)
            first = service.prepare_csv_cached(history_path, cache_dir=cache_dir)
            np.save(cache_dir / "features.npy", first.features[:, :1])

            second = service.prepare_csv_cached(history_path, cache_dir=cache_dir)

            self.assertEqual(second.summary.cache_status, "miss")
            self.assertEqual(second.features.shape[1], len(second.feature_columns))

    def test_invalid_cached_label_range_is_recomputed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "history.csv"
            cache_dir = Path(tmpdir) / "cache"
            sample_history().to_csv(history_path, index=False)
            service = DatasetService(self.config)
            first = service.prepare_csv_cached(history_path, cache_dir=cache_dir)
            labels = first.labels.copy()
            labels[0] = 9
            np.save(cache_dir / "labels.npy", labels)

            second = service.prepare_csv_cached(history_path, cache_dir=cache_dir)

            self.assertEqual(second.summary.cache_status, "miss")
            self.assertTrue(set(second.labels.tolist()).issubset({0, 1, 2}))

    def test_cached_array_dtype_mismatch_is_recomputed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "history.csv"
            cache_dir = Path(tmpdir) / "cache"
            sample_history().to_csv(history_path, index=False)
            service = DatasetService(self.config)
            first = service.prepare_csv_cached(history_path, cache_dir=cache_dir)
            np.save(cache_dir / "features.npy", first.features.astype(np.float64))

            second = service.prepare_csv_cached(history_path, cache_dir=cache_dir)

            self.assertEqual(second.summary.cache_status, "miss")
            self.assertEqual(second.features.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
