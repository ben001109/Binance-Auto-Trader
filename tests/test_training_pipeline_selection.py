import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from bat.research_config import DataConfig, ResearchConfig, TrainingConfig
from bat.data.features import DataQualityReport
from bat.training import _prepare_training_inputs
from bat.training import _resolve_training_model_config


class TrainingPipelineSelectionTest(unittest.TestCase):
    def test_research_pipeline_uses_dataset_service_outputs(self):
        config = ResearchConfig(
            data=DataConfig(seq_len=4),
            training=TrainingConfig(data_pipeline="research"),
        )
        prepared = SimpleNamespace(
            features=np.ones((8, 2), dtype=np.float32),
            labels=np.array([0, 1, 2, 1, 0, 2, 1, 1], dtype=np.int64),
            future_returns=np.linspace(0.01, 0.08, 8),
            feature_columns=["a", "b"],
            summary=SimpleNamespace(rows=8),
        )
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=8, freq="1h", tz="UTC"),
                "open": range(8),
                "high": range(8),
                "low": range(8),
                "close": range(8),
                "volume": range(8),
            }
        )

        with patch("bat.training.DatasetService") as service_class:
            service_class.return_value.prepare_dataframe.return_value = prepared

            inputs = _prepare_training_inputs(
                data_path="data/history.csv",
                df=df,
                data_pipeline="research",
                research_config=config,
            )

        service_class.assert_called_once_with(config)
        self.assertEqual(inputs.pipeline, "research")
        self.assertEqual(inputs.seq_length, 4)
        self.assertEqual(inputs.feature_cols, ["a", "b"])
        self.assertEqual(inputs.data_scaled.shape, (8, 2))
        self.assertTrue(np.array_equal(inputs.target_scaled, prepared.labels))
        self.assertTrue(np.array_equal(inputs.target_ret, prepared.future_returns))

    def test_training_config_defaults_to_legacy_pipeline(self):
        config = ResearchConfig()

        self.assertEqual(config.training.data_pipeline, "legacy")

    def test_research_pipeline_can_use_preprocessing_cache(self):
        config = ResearchConfig(
            data=DataConfig(seq_len=4),
            training=TrainingConfig(data_pipeline="research", use_preprocessing_cache=True),
        )
        prepared = SimpleNamespace(
            features=np.ones((8, 2), dtype=np.float32),
            labels=np.array([0, 1, 2, 1, 0, 2, 1, 1], dtype=np.int64),
            future_returns=np.linspace(0.01, 0.08, 8),
            feature_columns=["a", "b"],
            quality_report=DataQualityReport(
                rows_before=8,
                rows_after=8,
                duplicate_timestamps=0,
                dropped_nan_rows=0,
                missing_candles=0,
                nan_count=0,
                start_time="2024-01-01T00:00:00+00:00",
                end_time="2024-01-01T07:00:00+00:00",
            ),
            summary=SimpleNamespace(rows=8, cache_status="hit"),
        )

        with patch("bat.training.DatasetService") as service_class:
            service_class.return_value.prepare_csv_cached.return_value = prepared

            inputs = _prepare_training_inputs(
                data_path="data/history.csv",
                df=None,
                data_pipeline="research",
                research_config=config,
            )

        service_class.return_value.prepare_csv_cached.assert_called_once()
        self.assertEqual(inputs.pipeline, "research")
        self.assertEqual(inputs.last_trained_timestamp, 1704092400000)

    def test_training_model_override_updates_research_config(self):
        config = ResearchConfig()

        resolved = _resolve_training_model_config(config, model_name="cnn_lstm")

        self.assertEqual(resolved.model.name, "cnn_lstm")
        self.assertEqual(config.model.name, "cnn_lstm")


if __name__ == "__main__":
    unittest.main()
