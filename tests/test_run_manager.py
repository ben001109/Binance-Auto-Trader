import json
import pickle
import tempfile
import unittest
from pathlib import Path

import torch
import yaml

from bat.research_config import DataConfig, ModelConfig, ResearchConfig
from bat.services.run_manager import RunManager


class RunManagerTest(unittest.TestCase):
    def test_create_run_writes_required_artifacts(self):
        config = ResearchConfig(
            data=DataConfig(symbol="ETHUSDT", interval="1h"),
            model=ModelConfig(name="cnn_lstm"),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            run = RunManager(root=tmpdir).create_run(
                config=config,
                model_name="cnn_lstm",
                feature_columns=["log_return_1", "RSI_14"],
                label_distribution={0: 3, 1: 5, 2: 2},
                timestamp="20260102_030405",
            )

            self.assertEqual(run.path.name, "20260102_030405_cnn_lstm_ETHUSDT_1h")
            expected_files = {
                "config.yaml",
                "training_log.csv",
                "metrics.json",
                "report.json",
                "feature_columns.json",
                "label_distribution.json",
            }
            self.assertTrue(expected_files.issubset({path.name for path in run.path.iterdir()}))

            config_payload = yaml.safe_load((run.path / "config.yaml").read_text(encoding="utf-8"))
            self.assertEqual(config_payload["data"]["symbol"], "ETHUSDT")
            self.assertEqual(config_payload["model"]["name"], "cnn_lstm")

            log_header = (run.path / "training_log.csv").read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("epoch", log_header)
            self.assertIn("train_loss", log_header)

            features = json.loads((run.path / "feature_columns.json").read_text(encoding="utf-8"))
            labels = json.loads((run.path / "label_distribution.json").read_text(encoding="utf-8"))
            self.assertEqual(features, ["log_return_1", "RSI_14"])
            self.assertEqual(labels, {"0": 3, "1": 5, "2": 2})

    def test_save_best_model_writes_pt_file(self):
        model = torch.nn.Linear(2, 3)
        with tempfile.TemporaryDirectory() as tmpdir:
            run = RunManager(root=tmpdir).create_run(
                config=ResearchConfig(),
                model_name="lstm",
                feature_columns=["a", "b"],
                label_distribution={0: 1, 1: 1, 2: 1},
                timestamp="20260102_030405",
            )

            run.save_best_model(model)

            self.assertTrue((run.path / "best_model.pt").exists())

    def test_save_sklearn_model_writes_pkl_file(self):
        model = {"kind": "histgb"}
        with tempfile.TemporaryDirectory() as tmpdir:
            run = RunManager(root=tmpdir).create_run(
                config=ResearchConfig(),
                model_name="histgb",
                feature_columns=["a", "b"],
                label_distribution={0: 1, 1: 1, 2: 1},
                timestamp="20260102_030405",
            )

            run.save_sklearn_model(model)

            self.assertTrue((run.path / "best_model.pkl").exists())
            loaded = pickle.loads((run.path / "best_model.pkl").read_bytes())
            self.assertEqual(loaded, model)

    def test_write_epoch_metrics_appends_log_and_updates_metrics_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run = RunManager(root=tmpdir).create_run(
                config=ResearchConfig(),
                model_name="lstm",
                feature_columns=["a", "b"],
                label_distribution={0: 1, 1: 1, 2: 1},
                timestamp="20260102_030405",
            )

            run.write_epoch_metrics(
                {
                    "epoch": 1,
                    "train_loss": 0.5,
                    "val_loss": 0.4,
                    "val_accuracy": 0.75,
                    "val_macro_f1": 0.7,
                    "expected_return_after_cost": 0.01,
                    "win_rate": 0.6,
                    "profit_factor": 1.4,
                    "max_drawdown": -0.05,
                }
            )

            log_lines = (run.path / "training_log.csv").read_text(encoding="utf-8").splitlines()
            latest_metrics = json.loads((run.path / "metrics.json").read_text(encoding="utf-8"))
            self.assertEqual(len(log_lines), 2)
            self.assertIn("0.5", log_lines[1])
            self.assertEqual(latest_metrics["latest"]["val_accuracy"], 0.75)


if __name__ == "__main__":
    unittest.main()
