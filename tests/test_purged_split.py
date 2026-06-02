import unittest

import pandas as pd

from bat.data.split import purged_walk_forward_indices, walk_forward_splits


class PurgedSplitTest(unittest.TestCase):
    def test_purged_walk_forward_indices_leave_horizon_gap(self):
        folds = purged_walk_forward_indices(
            length=40,
            train_size=10,
            valid_size=5,
            step_size=5,
            purge_size=3,
            embargo_size=1,
        )

        self.assertGreaterEqual(len(folds), 2)
        for fold in folds:
            gap = min(fold.valid_indices) - max(fold.train_indices) - 1
            self.assertGreaterEqual(gap, 4)
            self.assertTrue(set(fold.train_indices).isdisjoint(fold.valid_indices))

    def test_time_walk_forward_supports_purge_and_embargo_windows(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=20, freq="1D", tz="UTC"),
                "close": range(20),
            }
        )

        folds = walk_forward_splits(
            df,
            train_window="5d",
            valid_window="3d",
            step_window="3d",
            purge_window="2d",
            embargo_window="1d",
        )

        self.assertGreaterEqual(len(folds), 2)
        for fold in folds:
            self.assertGreaterEqual(fold.valid_start - fold.train_end, pd.Timedelta("3d"))
            self.assertLess(max(fold.train_indices), min(fold.valid_indices))


if __name__ == "__main__":
    unittest.main()
