import unittest

from bat.services.dataset_service import DatasetSummary
from bat.tui.app import format_dataset_summary


class TuiDatasetStatusTest(unittest.TestCase):
    def test_format_dataset_summary_includes_distribution_and_warnings(self):
        summary = DatasetSummary(
            rows=100,
            start_time="2024-01-01T00:00:00+00:00",
            end_time="2024-01-02T00:00:00+00:00",
            missing_candles=2,
            nan_count=0,
            feature_count=14,
            selected_features=["log_return_1", "RSI_14"],
            label_distribution={0: 10, 1: 80, 2: 10},
            normalization_mode="rolling_zscore",
            seq_len=60,
            horizon=4,
            fee=0.001,
            slippage=0.0005,
            min_edge=0.003,
            warnings=["Label imbalance too high"],
            cache_status="hit",
        )

        text = format_dataset_summary(summary)

        self.assertIn("rows=100", text)
        self.assertIn("features=14", text)
        self.assertIn("cache=hit", text)
        self.assertIn("labels SELL/HOLD/BUY=10/80/10", text)
        self.assertIn("Label imbalance too high", text)

    def test_tui_has_dataset_check_entrypoint(self):
        source = "src/bat/tui/app.py"
        with open(source, "r", encoding="utf-8") as handle:
            content = handle.read()

        self.assertIn("btn_check_dataset", content)
        self.assertIn("dataset_status", content)
        self.assertIn("action_check_dataset", content)

    def test_tui_has_training_pipeline_selector(self):
        source = "src/bat/tui/app.py"
        with open(source, "r", encoding="utf-8") as handle:
            content = handle.read()

        self.assertIn("select_data_pipeline", content)
        self.assertIn("_data_pipeline_value", content)
        self.assertIn("data_pipeline=", content)

    def test_tui_has_model_type_selector(self):
        source = "src/bat/tui/app.py"
        with open(source, "r", encoding="utf-8") as handle:
            content = handle.read()

        self.assertIn("select_model_type", content)
        self.assertIn("_model_type_value", content)
        self.assertIn("model_name=", content)

    def test_tui_model_selector_includes_histgb_baseline(self):
        source = "src/bat/tui/app.py"
        with open(source, "r", encoding="utf-8") as handle:
            content = handle.read()

        self.assertIn('("HistGB", "histgb")', content)
        self.assertIn('{"lstm", "cnn_lstm", "histgb"}', content)


if __name__ == "__main__":
    unittest.main()
