import unittest

import numpy as np
import pandas as pd

from bat.data.features import clean_ohlcv


def valid_ohlcv(rows: int = 12) -> pd.DataFrame:
    close = np.linspace(100.0, 102.0, rows)
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=rows, freq="15min", tz="UTC"),
            "open": close - 0.1,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.linspace(10.0, 11.0, rows),
        }
    )


class OhlcvQualityGateTest(unittest.TestCase):
    def test_accepts_valid_ohlcv(self):
        cleaned, report = clean_ohlcv(valid_ohlcv(), expected_interval="15min")

        self.assertEqual(len(cleaned), report.rows_after)
        self.assertEqual(report.nan_count, 0)

    def test_rejects_non_positive_prices(self):
        df = valid_ohlcv()
        df.loc[3, "close"] = 0.0

        with self.assertRaisesRegex(ValueError, "positive"):
            clean_ohlcv(df, expected_interval="15min")

    def test_rejects_non_finite_ohlcv_values(self):
        df = valid_ohlcv()
        df.loc[3, "high"] = np.inf

        with self.assertRaisesRegex(ValueError, "finite"):
            clean_ohlcv(df, expected_interval="15min")

    def test_rejects_nan_ohlcv_values(self):
        df = valid_ohlcv()
        df.loc[3, "low"] = np.nan

        with self.assertRaisesRegex(ValueError, "finite"):
            clean_ohlcv(df, expected_interval="15min")

    def test_rejects_high_below_low(self):
        df = valid_ohlcv()
        df.loc[3, "high"] = df.loc[3, "low"] - 1.0

        with self.assertRaisesRegex(ValueError, "high"):
            clean_ohlcv(df, expected_interval="15min")

    def test_rejects_open_or_close_outside_range(self):
        df = valid_ohlcv()
        df.loc[3, "open"] = df.loc[3, "high"] + 1.0

        with self.assertRaisesRegex(ValueError, "range"):
            clean_ohlcv(df, expected_interval="15min")

    def test_rejects_negative_volume(self):
        df = valid_ohlcv()
        df.loc[3, "volume"] = -1.0

        with self.assertRaisesRegex(ValueError, "volume"):
            clean_ohlcv(df, expected_interval="15min")

    def test_rejects_extreme_price_jump_outlier(self):
        df = valid_ohlcv()
        df.loc[6, "open"] = 1000.0
        df.loc[6, "high"] = 1010.0
        df.loc[6, "low"] = 990.0
        df.loc[6, "close"] = 1000.0

        with self.assertRaisesRegex(ValueError, "outlier"):
            clean_ohlcv(df, expected_interval="15min")


if __name__ == "__main__":
    unittest.main()
