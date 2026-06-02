import unittest

import numpy as np
import pandas as pd

from bat.data.dataset import ResearchSequenceDataset, build_cost_aware_labels
from bat.data.features import DEFAULT_RESEARCH_FEATURES, clean_ohlcv, add_research_features
from bat.data.normalization import rolling_zscore_normalize


def sample_ohlcv(rows: int = 80) -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=rows, freq="15min", tz="UTC")
    close = pd.Series(np.linspace(100.0, 120.0, rows))
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": np.linspace(10.0, 20.0, rows),
        }
    )


class DataPipelineTest(unittest.TestCase):
    def test_clean_ohlcv_sorts_deduplicates_and_reports_gaps(self):
        df = pd.DataFrame(
            {
                "timestamp": [
                    "2024-01-01 00:15:00+00:00",
                    "2024-01-01 00:00:00+00:00",
                    "2024-01-01 00:15:00+00:00",
                    "2024-01-01 00:45:00+00:00",
                    "2024-01-01 01:00:00+00:00",
                ],
                "open": [2, 1, 2, 3, 4],
                "high": [2, 1, 2, 3, 4],
                "low": [2, 1, 2, 3, 4],
                "close": [2, 1, 2, 3, 4],
                "volume": [2, 1, 2, 3, 4],
            }
        )

        cleaned, report = clean_ohlcv(df, expected_interval="15min")

        self.assertEqual(cleaned["timestamp"].tolist(), sorted(cleaned["timestamp"].tolist()))
        self.assertEqual(report.rows_before, 5)
        self.assertEqual(report.duplicate_timestamps, 1)
        self.assertEqual(report.dropped_nan_rows, 0)
        self.assertEqual(report.missing_candles, 1)
        self.assertEqual(len(cleaned), 4)

    def test_add_research_features_creates_expected_columns(self):
        df = sample_ohlcv()

        featured = add_research_features(df)

        for column in DEFAULT_RESEARCH_FEATURES:
            self.assertIn(column, featured.columns)

    def test_rolling_zscore_uses_only_prior_window(self):
        df = pd.DataFrame({"feature": [1.0, 2.0, 100.0, 4.0]})

        normalized = rolling_zscore_normalize(df, ["feature"], window=2)

        expected_at_2 = (100.0 - 1.5) / (np.std([1.0, 2.0], ddof=1) + 1e-8)
        self.assertTrue(np.isnan(normalized.loc[0, "feature"]))
        self.assertAlmostEqual(normalized.loc[2, "feature"], expected_at_2)

    def test_cost_aware_labels_use_fee_slippage_and_min_edge(self):
        df = pd.DataFrame({"close": [100.0, 101.0, 100.0, 98.0, 98.0]})

        labels = build_cost_aware_labels(
            df,
            horizon=1,
            fee=0.001,
            slippage=0.001,
            min_edge=0.003,
        )

        self.assertEqual(labels.iloc[0], 2)
        self.assertEqual(labels.iloc[1], 0)
        self.assertEqual(labels.iloc[3], 1)
        self.assertTrue(pd.isna(labels.iloc[-1]))

    def test_research_sequence_dataset_returns_lstm_shape(self):
        features = np.arange(24, dtype=np.float32).reshape(8, 3)
        labels = np.array([1, 1, 2, 0, 1, 2, 0, 1], dtype=np.int64)

        dataset = ResearchSequenceDataset(features, labels, seq_len=4)
        x, y = dataset[1]

        self.assertEqual(len(dataset), 5)
        self.assertEqual(tuple(x.shape), (4, 3))
        self.assertEqual(y.item(), labels[4])


if __name__ == "__main__":
    unittest.main()
