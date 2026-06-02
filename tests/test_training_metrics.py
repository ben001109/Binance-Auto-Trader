import unittest

import numpy as np

from bat.training_metrics import (
    classification_report,
    distribution,
    max_drawdown,
    trading_report,
)


class TrainingMetricsTest(unittest.TestCase):
    def test_distribution_counts_all_three_classes(self):
        self.assertEqual(distribution([0, 1, 1, 2, 2, 2]), {0: 1, 1: 2, 2: 3})

    def test_classification_report_computes_accuracy_macro_f1_and_distributions(self):
        report = classification_report(
            y_true=np.array([0, 1, 2, 2]),
            y_pred=np.array([0, 1, 1, 2]),
        )

        self.assertAlmostEqual(report["accuracy"], 0.75)
        self.assertAlmostEqual(report["macro_f1"], (1.0 + (2 / 3) + (2 / 3)) / 3)
        self.assertEqual(report["label_distribution"], {0: 1, 1: 1, 2: 2})
        self.assertEqual(report["prediction_distribution"], {0: 1, 1: 2, 2: 1})

    def test_trading_report_accounts_for_direction_cost_and_profit_factor(self):
        report = trading_report(
            predictions=np.array([2, 0, 1, 2]),
            future_returns=np.array([0.02, -0.01, 0.03, -0.02]),
            fee=0.001,
            slippage=0.001,
        )

        self.assertEqual(report["trade_count"], 3)
        self.assertAlmostEqual(report["expected_return_after_cost"], (0.018 + 0.008 - 0.022) / 3)
        self.assertAlmostEqual(report["win_rate"], 2 / 3)
        self.assertAlmostEqual(report["profit_factor"], (0.018 + 0.008) / 0.022)
        self.assertLess(report["max_drawdown"], 0.0)

    def test_max_drawdown_uses_equity_curve(self):
        dd = max_drawdown([100.0, 110.0, 99.0, 120.0])

        self.assertAlmostEqual(dd, -0.1)


if __name__ == "__main__":
    unittest.main()
