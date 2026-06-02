import unittest

import pandas as pd

from bat.data.split import walk_forward_splits


class DataSplitTest(unittest.TestCase):
    def test_walk_forward_splits_are_time_ordered_and_non_overlapping(self):
        df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="1D", tz="UTC"),
                "close": range(10),
            }
        )

        folds = walk_forward_splits(
            df,
            train_window="3d",
            valid_window="2d",
            step_window="2d",
        )

        self.assertGreaterEqual(len(folds), 2)
        for fold in folds:
            self.assertLess(max(fold.train_indices), min(fold.valid_indices))
            self.assertTrue(set(fold.train_indices).isdisjoint(fold.valid_indices))
        self.assertLess(folds[0].train_start, folds[1].train_start)


if __name__ == "__main__":
    unittest.main()
