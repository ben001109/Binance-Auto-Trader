import unittest

import numpy as np
import pandas as pd

from bat.research_config import ResearchConfig, DataConfig, LabelConfig
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


class DatasetServiceTest(unittest.TestCase):
    def test_prepare_dataframe_returns_sequence_dataset_and_summary(self):
        config = ResearchConfig(
            data=DataConfig(seq_len=8, horizon=2, rolling_norm_window=12),
            label=LabelConfig(fee=0.001, slippage=0.0005, min_edge=0.001),
        )
        service = DatasetService(config)

        prepared = service.prepare_dataframe(sample_history())

        self.assertGreater(len(prepared.dataset), 0)
        self.assertEqual(prepared.features.shape[1], len(prepared.feature_columns))
        self.assertEqual(prepared.summary.seq_len, 8)
        self.assertEqual(prepared.summary.horizon, 2)
        self.assertEqual(prepared.summary.normalization_mode, "rolling_zscore")
        self.assertEqual(sum(prepared.summary.label_distribution.values()), len(prepared.labels))
        self.assertTrue(hasattr(prepared, "future_returns"))
        self.assertEqual(len(prepared.future_returns), len(prepared.labels))
        self.assertGreater(float(prepared.future_returns.max()), 0.0)

    def test_summary_warns_when_hold_label_is_too_high(self):
        config = ResearchConfig(
            data=DataConfig(seq_len=4, horizon=1, rolling_norm_window=5),
            label=LabelConfig(fee=0.1, slippage=0.1, min_edge=0.1),
        )
        service = DatasetService(config)

        prepared = service.prepare_dataframe(sample_history())

        self.assertIn("Label imbalance too high", prepared.summary.warnings)


if __name__ == "__main__":
    unittest.main()
