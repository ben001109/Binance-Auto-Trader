import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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


class DatasetCacheTest(unittest.TestCase):
    def test_prepare_csv_cached_writes_and_reuses_cache(self):
        config = ResearchConfig(
            data=DataConfig(seq_len=8, horizon=2, rolling_norm_window=12),
            label=LabelConfig(fee=0.001, slippage=0.0005, min_edge=0.001),
            training=TrainingConfig(use_preprocessing_cache=True),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "history.csv"
            cache_dir = Path(tmpdir) / "cache"
            sample_history().to_csv(history_path, index=False)
            service = DatasetService(config)

            first = service.prepare_csv_cached(history_path, cache_dir=cache_dir)

            self.assertEqual(first.summary.cache_status, "miss")
            self.assertTrue((cache_dir / "features.npy").exists())
            self.assertTrue((cache_dir / "labels.npy").exists())
            self.assertTrue((cache_dir / "future_returns.npy").exists())
            self.assertTrue((cache_dir / "meta.json").exists())

            with patch(
                "bat.services.dataset_service.add_research_features",
                side_effect=AssertionError("feature engineering should not rerun"),
            ):
                second = service.prepare_csv_cached(history_path, cache_dir=cache_dir)

        self.assertEqual(second.summary.cache_status, "hit")
        self.assertTrue(np.array_equal(first.features, second.features))
        self.assertTrue(np.array_equal(first.labels, second.labels))
        self.assertTrue(np.array_equal(first.future_returns, second.future_returns))

    def test_prepare_csv_cached_invalidates_when_config_changes(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "history.csv"
            cache_dir = Path(tmpdir) / "cache"
            sample_history().to_csv(history_path, index=False)

            first_config = ResearchConfig(data=DataConfig(seq_len=8, horizon=2, rolling_norm_window=12))
            second_config = ResearchConfig(data=DataConfig(seq_len=8, horizon=3, rolling_norm_window=12))

            DatasetService(first_config).prepare_csv_cached(history_path, cache_dir=cache_dir)
            second = DatasetService(second_config).prepare_csv_cached(history_path, cache_dir=cache_dir)

        self.assertEqual(second.summary.cache_status, "miss")


if __name__ == "__main__":
    unittest.main()
