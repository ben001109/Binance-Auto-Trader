import unittest
import inspect

import numpy as np
import torch

from bat.data.dataset import TimeSeriesDataset
from bat.research_config import DataConfig, ResearchConfig, ValidationConfig
from bat.training import (
    _aggregate_fold_metrics,
    _non_overlapping_validation_folds,
    _time_ordered_train_valid_indices,
    _validation_folds,
    evaluate_model_on_dataset,
)


class EchoLastStepModel(torch.nn.Module):
    def forward(self, x):
        return x[:, -1, :]


class ValidationLogitModel(torch.nn.Module):
    def forward(self, x):
        return x[:, -1, :]


class ValidationEvaluationTest(unittest.TestCase):
    def test_time_ordered_validation_split_keeps_tail_for_validation(self):
        train_indices, valid_indices = _time_ordered_train_valid_indices(10, valid_ratio=0.2)

        self.assertEqual(train_indices, list(range(8)))
        self.assertEqual(valid_indices, [8, 9])

    def test_evaluate_model_on_dataset_returns_validation_and_trade_metrics(self):
        labels = np.array([0, 1, 2, 1, 0, 2], dtype=np.int64)
        data = np.eye(3, dtype=np.float32)[labels]
        dataset = TimeSeriesDataset(data, labels, seq_length=2)
        future_returns = np.array([0.0, 0.01, -0.02, 0.03, -0.04, 0.05], dtype=float)

        metrics = evaluate_model_on_dataset(
            model=EchoLastStepModel(),
            dataset=dataset,
            indices=list(range(len(dataset))),
            device=torch.device("cpu"),
            fee=0.001,
            slippage=0.001,
            future_returns=future_returns[1 : 1 + len(dataset)],
        )

        self.assertIn("val_loss", metrics)
        self.assertAlmostEqual(metrics["val_accuracy"], 1.0)
        self.assertAlmostEqual(metrics["val_macro_f1"], 1.0)
        self.assertEqual(metrics["prediction_distribution"], {0: 1, 1: 2, 2: 1})
        self.assertEqual(metrics["trade_count"], 2)
        self.assertIn("expected_return_after_cost", metrics)

    def test_decision_filter_turns_low_confidence_actions_into_hold(self):
        update = getattr(__import__("bat.training", fromlist=["_apply_decision_filter"]), "_apply_decision_filter", None)
        self.assertIsNotNone(update, "_apply_decision_filter should exist")
        logits = torch.tensor(
            [
                [0.2, 0.0, 0.3],
                [0.3, 0.0, 0.2],
                [2.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )

        predictions, filtered_count = update(logits, confidence_threshold=0.6)

        self.assertEqual(predictions.tolist(), [1, 1, 0])
        self.assertEqual(filtered_count, 2)

    def test_decision_filter_turns_low_edge_actions_into_hold(self):
        update = getattr(__import__("bat.training", fromlist=["_apply_decision_filter"]), "_apply_decision_filter", None)
        self.assertIsNotNone(update, "_apply_decision_filter should exist")
        self.assertIn(
            "edge_threshold",
            inspect.signature(update).parameters,
            "_apply_decision_filter should accept edge_threshold",
        )
        probs = torch.tensor(
            [
                [0.40, 0.25, 0.35],
                [0.70, 0.10, 0.20],
                [0.35, 0.25, 0.40],
                [0.30, 0.40, 0.30],
            ],
            dtype=torch.float32,
        )

        predictions, filtered_count = update(
            torch.log(probs),
            confidence_threshold=0.0,
            edge_threshold=0.10,
        )

        self.assertEqual(predictions.tolist(), [1, 0, 1, 1])
        self.assertEqual(filtered_count, 2)

    def test_evaluate_model_on_dataset_applies_confidence_filter_to_trade_metrics(self):
        signature = inspect.signature(evaluate_model_on_dataset)
        self.assertIn(
            "confidence_threshold",
            signature.parameters,
            "evaluate_model_on_dataset should accept confidence_threshold",
        )
        self.assertIn(
            "edge_threshold",
            signature.parameters,
            "evaluate_model_on_dataset should accept edge_threshold",
        )
        logits = np.array(
            [
                [0.2, 0.0, 0.3],
                [0.3, 0.0, 0.2],
                [2.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
            ],
            dtype=np.float32,
        )
        labels = np.array([2, 0, 0, 1], dtype=np.int64)
        dataset = TimeSeriesDataset(logits, labels, seq_length=1)

        metrics = evaluate_model_on_dataset(
            model=ValidationLogitModel(),
            dataset=dataset,
            indices=list(range(len(dataset))),
            device=torch.device("cpu"),
            fee=0.001,
            slippage=0.001,
            future_returns=np.array([0.02, -0.02, -0.01], dtype=float),
            confidence_threshold=0.6,
        )

        self.assertEqual(metrics["prediction_distribution"], {0: 1, 1: 2, 2: 0})
        self.assertEqual(metrics["trade_count"], 1)
        self.assertEqual(metrics["filtered_to_hold"], 2)
        self.assertEqual(metrics["decision_confidence_threshold"], 0.6)

    def test_evaluate_model_on_dataset_applies_edge_filter_to_trade_metrics(self):
        self.assertIn(
            "edge_threshold",
            inspect.signature(evaluate_model_on_dataset).parameters,
            "evaluate_model_on_dataset should accept edge_threshold",
        )
        probs = np.array(
            [
                [0.40, 0.25, 0.35],
                [0.70, 0.10, 0.20],
                [0.35, 0.25, 0.40],
                [0.30, 0.40, 0.30],
            ],
            dtype=np.float32,
        )
        labels = np.array([0, 0, 2, 1], dtype=np.int64)
        dataset = TimeSeriesDataset(np.log(probs), labels, seq_length=1)

        metrics = evaluate_model_on_dataset(
            model=ValidationLogitModel(),
            dataset=dataset,
            indices=list(range(len(dataset))),
            device=torch.device("cpu"),
            fee=0.001,
            slippage=0.001,
            future_returns=np.array([-0.01, -0.02, 0.02], dtype=float),
            confidence_threshold=0.0,
            edge_threshold=0.10,
        )

        self.assertEqual(metrics["prediction_distribution"], {0: 1, 1: 2, 2: 0})
        self.assertEqual(metrics["trade_count"], 1)
        self.assertEqual(metrics["filtered_to_hold"], 2)
        self.assertEqual(metrics["decision_edge_threshold"], 0.10)

    def test_validation_config_defaults_to_no_edge_filter(self):
        config = ValidationConfig()
        self.assertTrue(hasattr(config, "decision_edge_threshold"))
        self.assertAlmostEqual(config.decision_edge_threshold, 0.0)

    def test_validation_folds_honor_walk_forward_with_horizon_purge(self):
        config = ResearchConfig(
            data=DataConfig(horizon=3),
            validation=ValidationConfig(method="walk_forward", holdout_ratio=0.2),
        )

        folds = _validation_folds(40, config)

        self.assertGreaterEqual(len(folds), 2)
        for fold in folds:
            gap = min(fold.valid_indices) - max(fold.train_indices) - 1
            self.assertGreaterEqual(gap, 6)

    def test_neural_validation_eval_folds_do_not_overlap_actual_train_indices(self):
        config = ResearchConfig(
            data=DataConfig(horizon=2),
            validation=ValidationConfig(method="walk_forward", holdout_ratio=0.2),
        )
        folds = _validation_folds(100, config)
        train_indices = folds[-1].train_indices

        eval_folds = _non_overlapping_validation_folds(train_indices, folds)

        self.assertGreaterEqual(len(eval_folds), 1)
        for fold in eval_folds:
            self.assertTrue(set(train_indices).isdisjoint(fold.valid_indices))

    def test_walk_forward_fallback_preserves_horizon_gap(self):
        config = ResearchConfig(
            data=DataConfig(horizon=3),
            validation=ValidationConfig(method="walk_forward", holdout_ratio=0.25),
        )

        folds = _validation_folds(8, config)

        if folds:
            gap = min(folds[0].valid_indices) - max(folds[0].train_indices) - 1
            self.assertGreaterEqual(gap, 6)

    def test_aggregate_fold_metrics_combines_counts_and_distributions(self):
        metrics = _aggregate_fold_metrics(
            [
                {
                    "val_loss": 0.4,
                    "trade_count": 10,
                    "filtered_to_hold": 2,
                    "max_drawdown": -0.02,
                    "prediction_distribution": {0: 2, 1: 6, 2: 2},
                },
                {
                    "val_loss": 0.6,
                    "trade_count": 20,
                    "filtered_to_hold": 3,
                    "max_drawdown": -0.08,
                    "prediction_distribution": {0: 3, 1: 7, 2: 10},
                },
            ]
        )

        self.assertEqual(metrics["fold_count"], 2)
        self.assertAlmostEqual(metrics["val_loss"], 0.5)
        self.assertEqual(metrics["trade_count"], 30)
        self.assertEqual(metrics["filtered_to_hold"], 5)
        self.assertEqual(metrics["max_drawdown"], -0.08)
        self.assertEqual(metrics["prediction_distribution"], {0: 5, 1: 13, 2: 12})


if __name__ == "__main__":
    unittest.main()
