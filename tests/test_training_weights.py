import unittest

import numpy as np

from bat import training
from bat.research_config import TrainingConfig


class TrainingWeightsTest(unittest.TestCase):
    def _compute(self, targets, returns, strength):
        compute = getattr(training, "_compute_training_weights", None)
        self.assertIsNotNone(compute, "_compute_training_weights should exist")
        return compute(
            np.asarray(targets, dtype=np.int64),
            np.asarray(returns, dtype=float),
            class_weight_strength=strength,
        )

    def test_zero_class_weight_strength_keeps_class_weights_neutral(self):
        targets = [0] * 10 + [1] * 80 + [2] * 10
        returns = np.zeros(len(targets), dtype=float)

        weights = self._compute(targets, returns, strength=0.0)

        self.assertTrue(np.allclose(weights, np.ones(len(targets))))

    def test_partial_class_weight_strength_reduces_minority_amplification(self):
        targets = [0] * 10 + [1] * 80 + [2] * 10
        returns = np.zeros(len(targets), dtype=float)

        full = self._compute(targets, returns, strength=1.0)
        soft = self._compute(targets, returns, strength=0.35)

        self.assertGreater(full[0], soft[0])
        self.assertGreater(soft[0], 1.0)
        self.assertLess(full[10], soft[10])
        self.assertLess(soft[10], 1.0)

    def test_training_config_defaults_to_full_class_weighting(self):
        config = TrainingConfig()
        self.assertTrue(hasattr(config, "class_weight_strength"))
        self.assertAlmostEqual(config.class_weight_strength, 1.0)


if __name__ == "__main__":
    unittest.main()
