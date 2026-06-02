import unittest

from bat import training


class EarlyStoppingTest(unittest.TestCase):
    def _update(self, state, metrics, *, epoch=1, metric="expected_return_after_cost", patience=2):
        update = getattr(training, "_update_early_stopping", None)
        self.assertIsNotNone(update, "_update_early_stopping should exist")
        return update(
            state,
            metrics,
            epoch=epoch,
            metric_name=metric,
            patience=patience,
        )

    def test_first_metric_sets_best_without_stopping(self):
        state = self._update(None, {"expected_return_after_cost": 0.01}, epoch=1)

        self.assertEqual(state.best_epoch, 1)
        self.assertAlmostEqual(state.best_metric, 0.01)
        self.assertTrue(state.improved)
        self.assertFalse(state.should_stop)
        self.assertEqual(state.bad_epochs, 0)
        self.assertIsNone(state.stop_reason)

    def test_patience_exhaustion_stops_after_metric_stalls(self):
        state = self._update(None, {"expected_return_after_cost": 0.01}, epoch=1, patience=2)
        state = self._update(state, {"expected_return_after_cost": 0.009}, epoch=2, patience=2)
        state = self._update(state, {"expected_return_after_cost": 0.008}, epoch=3, patience=2)

        self.assertEqual(state.best_epoch, 1)
        self.assertAlmostEqual(state.best_metric, 0.01)
        self.assertEqual(state.bad_epochs, 2)
        self.assertTrue(state.should_stop)
        self.assertEqual(
            state.stop_reason,
            "expected_return_after_cost did not improve for 2 epochs",
        )

    def test_missing_metric_does_not_consume_patience(self):
        state = self._update(None, {"expected_return_after_cost": 0.01}, epoch=1, patience=1)
        state = self._update(state, {"val_loss": 0.5}, epoch=2, patience=1)

        self.assertEqual(state.best_epoch, 1)
        self.assertEqual(state.bad_epochs, 0)
        self.assertFalse(state.should_stop)
        self.assertEqual(state.stop_reason, "metric expected_return_after_cost unavailable")

    def test_val_loss_uses_lower_value_as_improvement(self):
        state = self._update(None, {"val_loss": 0.5}, epoch=1, metric="val_loss", patience=1)
        state = self._update(state, {"val_loss": 0.4}, epoch=2, metric="val_loss", patience=1)

        self.assertEqual(state.best_epoch, 2)
        self.assertAlmostEqual(state.best_metric, 0.4)
        self.assertTrue(state.improved)
        self.assertFalse(state.should_stop)
        self.assertEqual(state.bad_epochs, 0)


if __name__ == "__main__":
    unittest.main()
