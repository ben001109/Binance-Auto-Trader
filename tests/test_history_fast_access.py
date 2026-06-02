import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bat.tui.app import CryptoApp


class HistoryFastAccessTest(unittest.TestCase):
    def test_latest_timestamp_reads_tail_without_full_csv_scan(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "history.csv"
            path.write_text(
                "timestamp,open,high,low,close,volume\n"
                "0,1,1,1,1,1\n"
                "2017-11-06 03:54:00,1,1,1,1,1\n"
                "1779694380000,1,1,1,1,1\n",
                encoding="utf-8",
            )

            with patch("csv.reader", side_effect=AssertionError("full csv scan not allowed")):
                latest = CryptoApp._get_latest_timestamp(None, str(path))

        self.assertEqual(latest, 1779694380000.0)

    def test_history_count_uses_metadata_row_count_when_available(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "history.csv"
            meta_path = Path(tmpdir) / "history.meta.json"
            path.write_text("timestamp,open\n0,1\n", encoding="utf-8")
            meta_path.write_text(
                json.dumps(
                    {
                        "version": 1,
                        "symbol": "BTCUSDT",
                        "interval": "1m",
                        "row_count": 4_603_977,
                        "latest_timestamp": 1779694380000,
                    }
                ),
                encoding="utf-8",
            )

            count = CryptoApp._history_count(None, str(path))

        self.assertEqual(count, 4_603_977)


if __name__ == "__main__":
    unittest.main()
