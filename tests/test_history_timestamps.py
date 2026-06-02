import unittest

import pandas as pd

from bat.data.integrity import check_data_gaps
from bat.data.timestamps import parse_timestamp_ms, normalize_timestamp_series
from bat.training import _latest_timestamp_from_df


class HistoryTimestampTest(unittest.TestCase):
    def test_parse_timestamp_accepts_datetime_string_and_milliseconds(self):
        self.assertEqual(parse_timestamp_ms("2017-11-06 03:54:00"), 1509940440000)
        self.assertEqual(parse_timestamp_ms("1779694380000"), 1779694380000)
        self.assertEqual(parse_timestamp_ms(1779694380000), 1779694380000)

    def test_parse_timestamp_rejects_empty_values(self):
        for value in (None, "", pd.NaT):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    parse_timestamp_ms(value)

    def test_normalize_timestamp_series_handles_mixed_history_values(self):
        values = pd.Series(["2017-11-06 03:54:00", "1779694380000", 1779694380000])

        normalized = normalize_timestamp_series(values)

        self.assertEqual(normalized.iloc[0].value // 1_000_000, 1509940440000)
        self.assertEqual(normalized.iloc[1].value // 1_000_000, 1779694380000)
        self.assertEqual(normalized.iloc[2].value // 1_000_000, 1779694380000)
        self.assertEqual(str(normalized.dt.tz), "UTC")

    def test_latest_timestamp_scans_unsorted_history(self):
        df = pd.DataFrame({"timestamp": [9_999_999] + list(range(2000))})

        self.assertEqual(_latest_timestamp_from_df(df), 9_999_999)

    def test_gap_check_normalizes_before_sorting_mixed_timestamps(self):
        interval_ms = 60_000
        base = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000) - interval_ms * 2
        middle = base + interval_ms
        latest = base + interval_ms * 2
        df = pd.DataFrame(
            {
                "timestamp": [
                    str(base),
                    pd.to_datetime(middle, unit="ms", utc=True).strftime("%Y-%m-%d %H:%M:%S"),
                    str(latest),
                ],
                "close": [1, 2, 3],
            }
        )

        self.assertEqual(check_data_gaps(df, interval_ms), [])


if __name__ == "__main__":
    unittest.main()
