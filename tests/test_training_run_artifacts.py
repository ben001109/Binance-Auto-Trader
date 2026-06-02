import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import torch

from bat.research_config import DataConfig, ModelConfig, ResearchConfig, TrainingConfig, ValidationConfig
from bat.services.run_manager import RunManager
from bat.training import TrainingInputs, _create_training_run, _save_run_best_model, train_model


class TinySequenceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(2, 3)

    def forward(self, x):
        return self.fc(x[:, -1, :])


class TrainingRunArtifactsTest(unittest.TestCase):
    def test_create_training_run_uses_feature_columns_and_label_distribution(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run = _create_training_run(
                config=ResearchConfig(),
                model_name="lstm",
                feature_cols=["a", "b"],
                targets=np.array([0, 1, 1, 2, 2, 2]),
                runs_root=tmpdir,
                timestamp="20260102_030405",
            )

            self.assertTrue((run.path / "config.yaml").exists())
            self.assertTrue((run.path / "label_distribution.json").exists())

    def test_save_run_best_model_writes_run_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run = _create_training_run(
                config=ResearchConfig(),
                model_name="lstm",
                feature_cols=["a", "b"],
                targets=np.array([0, 1, 2]),
                runs_root=tmpdir,
                timestamp="20260102_030405",
            )

            _save_run_best_model(run, torch.nn.Linear(2, 3))

            self.assertTrue((run.path / "best_model.pt").exists())

    def test_train_model_creates_run_and_saves_best_model(self):
        inputs = TrainingInputs(
            data_scaled=np.ones((8, 2), dtype=np.float32),
            target_scaled=np.array([0, 1, 2, 1, 0, 2, 1, 1], dtype=np.int64),
            df=pd.DataFrame(),
            target_ret=np.zeros(8),
            processor=None,
            feature_cols=["a", "b"],
            seq_length=2,
            last_trained_timestamp=123,
            pipeline="research",
        )

        logs = []
        with patch("bat.training._prepare_training_inputs", return_value=inputs), patch(
            "bat.training.create_model", return_value=TinySequenceModel()
        ), patch("bat.training._create_training_run") as create_run, patch(
            "bat.training._save_run_best_model"
        ) as save_best, patch("bat.training.torch.save"):
            run = SimpleNamespace(path="runs/test", write_epoch_metrics=Mock(), write_report=Mock())
            create_run.return_value = run

            train_model(
                epochs=1,
                research_config=ResearchConfig(),
                data_pipeline="research",
                on_log=logs.append,
            )

        create_run.assert_called_once()
        save_best.assert_called_once()
        run.write_epoch_metrics.assert_called()
        self.assertTrue(any("Run directory: runs/test" in item for item in logs))

    def test_train_model_stops_early_and_writes_report(self):
        inputs = TrainingInputs(
            data_scaled=np.ones((8, 2), dtype=np.float32),
            target_scaled=np.array([0, 1, 2, 1, 0, 2, 1, 1], dtype=np.int64),
            df=pd.DataFrame(),
            target_ret=np.zeros(8),
            processor=None,
            feature_cols=["a", "b"],
            seq_length=2,
            last_trained_timestamp=123,
            pipeline="research",
        )
        config = ResearchConfig(
            training=TrainingConfig(
                early_stopping_patience=1,
                early_stopping_metric="expected_return_after_cost",
            )
        )
        metrics = {
            "val_loss": 0.5,
            "val_accuracy": 0.5,
            "val_macro_f1": 0.4,
            "expected_return_after_cost": 0.01,
            "win_rate": 0.5,
            "profit_factor": 1.0,
            "max_drawdown": 0.0,
        }
        logs = []
        statuses = []

        with tempfile.TemporaryDirectory() as tmpdir:
            runs = []

            def create_run(config, model_name, feature_cols, targets):
                run = RunManager(root=tmpdir).create_run(
                    config=config,
                    model_name=model_name,
                    feature_columns=feature_cols,
                    label_distribution={0: 2, 1: 4, 2: 2},
                    timestamp="20260102_030405",
                )
                runs.append(run)
                return run

            with patch("bat.training._prepare_training_inputs", return_value=inputs), patch(
                "bat.training.create_model", return_value=TinySequenceModel()
            ), patch("bat.training._create_training_run", side_effect=create_run), patch(
                "bat.training.evaluate_model_on_dataset", return_value=metrics
            ), patch("bat.training._save_run_best_model") as save_best, patch(
                "bat.training.torch.save"
            ), patch(
                "bat.training.os.path.exists", return_value=False
            ):
                train_model(
                    epochs=5,
                    research_config=config,
                    data_pipeline="research",
                    on_log=logs.append,
                    on_status=statuses.append,
                )

            run = runs[0]
            log_lines = (run.path / "training_log.csv").read_text(encoding="utf-8").splitlines()
            report = json.loads((run.path / "report.json").read_text(encoding="utf-8"))

        self.assertEqual(len(log_lines), 3)
        self.assertEqual(save_best.call_count, 1)
        self.assertEqual(report["status"], "early_stopped")
        self.assertEqual(report["completed_epochs"], 2)
        self.assertEqual(report["best_epoch"], 1)
        self.assertAlmostEqual(report["best_metric"], 0.01)
        self.assertEqual(
            report["stop_reason"],
            "expected_return_after_cost did not improve for 1 epochs",
        )
        self.assertTrue(any("Early stopping" in item for item in logs))
        self.assertTrue(any(item.get("early_stopped") for item in statuses))

    def test_train_model_does_not_save_best_when_metric_unavailable(self):
        inputs = TrainingInputs(
            data_scaled=np.ones((8, 2), dtype=np.float32),
            target_scaled=np.array([0, 1, 2, 1, 0, 2, 1, 1], dtype=np.int64),
            df=pd.DataFrame(),
            target_ret=np.zeros(8),
            processor=None,
            feature_cols=["a", "b"],
            seq_length=2,
            last_trained_timestamp=123,
            pipeline="research",
        )
        config = ResearchConfig(
            training=TrainingConfig(
                early_stopping_patience=1,
                early_stopping_metric="missing_metric",
            )
        )
        metrics = {
            "val_loss": 0.5,
            "val_accuracy": 0.5,
            "val_macro_f1": 0.4,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            runs = []

            def create_run(config, model_name, feature_cols, targets):
                run = RunManager(root=tmpdir).create_run(
                    config=config,
                    model_name=model_name,
                    feature_columns=feature_cols,
                    label_distribution={0: 2, 1: 4, 2: 2},
                    timestamp="20260102_030405",
                )
                runs.append(run)
                return run

            with patch("bat.training._prepare_training_inputs", return_value=inputs), patch(
                "bat.training.create_model", return_value=TinySequenceModel()
            ), patch("bat.training._create_training_run", side_effect=create_run), patch(
                "bat.training.evaluate_model_on_dataset", return_value=metrics
            ), patch("bat.training._save_run_best_model") as save_best, patch(
                "bat.training.torch.save"
            ), patch(
                "bat.training.os.path.exists", return_value=False
            ):
                train_model(
                    epochs=1,
                    research_config=config,
                    data_pipeline="research",
                )

            report = json.loads((runs[0].path / "report.json").read_text(encoding="utf-8"))

        self.assertEqual(save_best.call_count, 0)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["best_epoch"], 0)
        self.assertIsNone(report["best_metric"])
        self.assertEqual(report["stop_reason"], "metric missing_metric unavailable")

    def test_train_model_reports_user_stopped_run(self):
        inputs = TrainingInputs(
            data_scaled=np.ones((8, 2), dtype=np.float32),
            target_scaled=np.array([0, 1, 2, 1, 0, 2, 1, 1], dtype=np.int64),
            df=pd.DataFrame(),
            target_ret=np.zeros(8),
            processor=None,
            feature_cols=["a", "b"],
            seq_length=2,
            last_trained_timestamp=123,
            pipeline="research",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            runs = []

            def create_run(config, model_name, feature_cols, targets):
                run = RunManager(root=tmpdir).create_run(
                    config=config,
                    model_name=model_name,
                    feature_columns=feature_cols,
                    label_distribution={0: 2, 1: 4, 2: 2},
                    timestamp="20260102_030405",
                )
                runs.append(run)
                return run

            with patch("bat.training._prepare_training_inputs", return_value=inputs), patch(
                "bat.training.create_model", return_value=TinySequenceModel()
            ), patch("bat.training._create_training_run", side_effect=create_run), patch(
                "bat.training.should_stop_training", return_value=True
            ), patch("bat.training._save_run_best_model") as save_best, patch(
                "bat.training.torch.save"
            ), patch(
                "bat.training.os.path.exists", return_value=False
            ):
                train_model(
                    epochs=5,
                    research_config=ResearchConfig(),
                    data_pipeline="research",
                )

            report = json.loads((runs[0].path / "report.json").read_text(encoding="utf-8"))

        self.assertEqual(save_best.call_count, 0)
        self.assertEqual(report["status"], "stopped")
        self.assertEqual(report["completed_epochs"], 0)
        self.assertEqual(report["stop_reason"], "stopped by user")

    def test_train_model_histgb_creates_pickle_model_artifacts(self):
        rows = 36
        features = np.column_stack(
            [
                np.linspace(-1.0, 1.0, rows),
                np.sin(np.linspace(0.0, 6.0, rows)),
            ]
        ).astype(np.float32)
        targets = np.asarray(([0, 1, 2] * 12), dtype=np.int64)
        inputs = TrainingInputs(
            data_scaled=features,
            target_scaled=targets,
            df=pd.DataFrame(),
            target_ret=np.linspace(-0.02, 0.02, rows),
            processor=None,
            feature_cols=["a", "b"],
            seq_length=2,
            last_trained_timestamp=123,
            pipeline="research",
        )
        config = ResearchConfig(
            data=DataConfig(seq_len=2),
            model=ModelConfig(name="histgb"),
            training=TrainingConfig(data_pipeline="research"),
            validation=ValidationConfig(
                holdout_ratio=0.25,
                decision_confidence_threshold=0.0,
                decision_edge_threshold=0.0,
            ),
        )
        logs = []

        with tempfile.TemporaryDirectory() as tmpdir:
            runs = []

            def create_run(config, model_name, feature_cols, targets):
                run = RunManager(root=tmpdir).create_run(
                    config=config,
                    model_name=model_name,
                    feature_columns=feature_cols,
                    label_distribution={0: 12, 1: 12, 2: 12},
                    timestamp="20260102_030405",
                )
                runs.append(run)
                return run

            with patch("bat.training._prepare_training_inputs", return_value=inputs), patch(
                "bat.training._create_training_run", side_effect=create_run
            ), patch("bat.training.os.path.exists", return_value=False):
                model, processor, df, last_ts = train_model(
                    epochs=3,
                    research_config=config,
                    data_pipeline="research",
                    model_name="histgb",
                    on_log=logs.append,
                )

            run = runs[0]
            log_lines = (run.path / "training_log.csv").read_text(encoding="utf-8").splitlines()
            report = json.loads((run.path / "report.json").read_text(encoding="utf-8"))
            artifact_exists = (run.path / "best_model.pkl").exists()

        self.assertIsNone(processor)
        self.assertEqual(last_ts, 123)
        self.assertTrue(artifact_exists)
        self.assertEqual(len(log_lines), 2)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["model_family"], "sklearn")
        self.assertTrue(any("model=histgb" in item for item in logs))


if __name__ == "__main__":
    unittest.main()
