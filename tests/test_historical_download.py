import unittest
import tempfile
import time
import json
import threading
from pathlib import Path
from unittest.mock import patch

from bat.execution import spot_client
from bat import simulation


class _FakeResponse:
    def __init__(self, rows):
        self._rows = rows

    def data(self):
        return self._rows


class _FlakyRestAPI:
    def __init__(self):
        self.calls = 0

    def klines(self, **kwargs):
        self.calls += 1
        start_time = kwargs["start_time"]
        if self.calls == 1:
            return _FakeResponse([[start_time], [start_time + 60_000]])
        if self.calls == 2:
            raise ConnectionError("temporary disconnect")
        return _FakeResponse([[start_time], [start_time + 60_000]])


class _FakeClient:
    def __init__(self):
        self.rest_api = _FlakyRestAPI()


class _HangingThenHealthyRestAPI:
    def __init__(self):
        self.calls = 0

    def klines(self, **kwargs):
        self.calls += 1
        start_time = kwargs["start_time"]
        if self.calls == 1:
            time.sleep(0.2)
        return _FakeResponse([[start_time], [start_time + 60_000]])


class _HangingThenHealthyClient:
    def __init__(self):
        self.rest_api = _HangingThenHealthyRestAPI()


class _OverlapDetectingRestAPI:
    def __init__(self):
        self.calls = 0
        self.active = False
        self.overlapped = False
        self.lock = threading.Lock()

    def klines(self, **kwargs):
        with self.lock:
            self.calls += 1
            call_number = self.calls
            if self.active:
                self.overlapped = True
            self.active = True
        try:
            if call_number == 1:
                time.sleep(0.1)
            start_time = kwargs["start_time"]
            return _FakeResponse([[start_time], [start_time + 60_000]])
        finally:
            with self.lock:
                self.active = False


class _OverlapDetectingClient:
    def __init__(self):
        self.rest_api = _OverlapDetectingRestAPI()


class HistoricalDownloadTest(unittest.IsolatedAsyncioTestCase):
    async def test_retries_transient_disconnect_and_continues_same_page(self):
        client = _FakeClient()

        with patch.object(spot_client, "KLINES_LIMIT", 2):
            rows = await spot_client.async_historical_klines(
                client,
                "BTCUSDT",
                "1m",
                "1970-01-01 00:00:00",
                "1970-01-01 00:04:00",
                max_retries=2,
                retry_delay=0,
            )

        self.assertEqual([row[0] for row in rows], [0, 60_000, 120_000, 180_000])
        self.assertEqual(client.rest_api.calls, 3)

    async def test_streams_pages_without_collecting_all_rows(self):
        client = _FakeClient()
        chunks = []

        async def on_chunk(rows, total, current_ms, end_ms):
            chunks.append((rows, total, current_ms, end_ms))

        with patch.object(spot_client, "KLINES_LIMIT", 2):
            rows = await spot_client.async_historical_klines(
                client,
                "BTCUSDT",
                "1m",
                "1970-01-01 00:00:00",
                "1970-01-01 00:04:00",
                on_chunk=on_chunk,
                collect=False,
                max_retries=2,
                retry_delay=0,
            )

        self.assertEqual(rows, [])
        self.assertEqual([[row[0] for row in chunk[0]] for chunk in chunks], [[0, 60_000], [120_000, 180_000]])
        self.assertEqual([chunk[1] for chunk in chunks], [2, 4])

    async def test_times_out_hung_request_and_retries_next_attempt(self):
        client = _HangingThenHealthyClient()

        with patch.object(spot_client, "KLINES_LIMIT", 2):
            rows = await spot_client.async_historical_klines(
                client,
                "BTCUSDT",
                "1m",
                "1970-01-01 00:00:00",
                "1970-01-01 00:02:00",
                max_retries=1,
                retry_delay=0,
                request_timeout=0.01,
            )

        self.assertEqual([row[0] for row in rows], [0, 60_000])
        self.assertEqual(client.rest_api.calls, 2)

    async def test_timeout_retry_does_not_overlap_inflight_request(self):
        client = _OverlapDetectingClient()

        with patch.object(spot_client, "KLINES_LIMIT", 2):
            rows = await spot_client.async_historical_klines(
                client,
                "BTCUSDT",
                "1m",
                "1970-01-01 00:00:00",
                "1970-01-01 00:02:00",
                max_retries=1,
                retry_delay=0,
                request_timeout=0.01,
            )

        self.assertEqual([row[0] for row in rows], [0, 60_000])
        self.assertFalse(client.rest_api.overlapped)


class HistoricalPersistenceTest(unittest.TestCase):
    def test_append_klines_persists_batch_without_rewriting_existing_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "history.csv"

            count = simulation.append_klines(path, [[0, "1"], [60_000, "2"]])
            count += simulation.append_klines(path, [[120_000, "3"]])

            lines = path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(count, 3)
        self.assertEqual(lines[0].split(","), simulation.KLINE_HEADERS)
        self.assertEqual([line.split(",")[0] for line in lines[1:]], ["0", "60000", "120000"])

    def test_history_metadata_detects_symbol_or_interval_mismatch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            meta_path = Path(tmpdir) / "history.meta.json"

            simulation.write_history_metadata(meta_path, "BTCUSDT", "1m")

            self.assertTrue(simulation.history_metadata_matches(meta_path, "BTCUSDT", "1m"))
            self.assertFalse(simulation.history_metadata_matches(meta_path, "BNBUSDT", "1m"))
            self.assertFalse(simulation.history_metadata_matches(meta_path, "BTCUSDT", "15m"))

    def test_write_klines_refreshes_existing_metadata_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "history.csv"
            meta_path = Path(tmpdir) / "history.meta.json"
            simulation.write_history_metadata(
                meta_path,
                "BTCUSDT",
                "1m",
                row_count=99,
                latest_timestamp=0,
            )

            count = simulation.write_klines(
                history_path,
                [[0, "1"], [60_000, "2"]],
                overwrite=True,
            )
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))

        self.assertEqual(count, 2)
        self.assertEqual(metadata["row_count"], 2)
        self.assertEqual(metadata["latest_timestamp"], 60_000)

    def test_write_klines_merge_normalizes_existing_timestamp_strings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "history.csv"
            history_path.write_text(
                "timestamp,open,high,low,close,volume,close_time,q_vol,trades,tb_base,tb_quote,ignore\n"
                "2017-11-06 03:54:00,1,1,1,1,1,,,,,,\n",
                encoding="utf-8",
            )

            simulation.write_klines(
                history_path,
                [[1779694380000, "2", "2", "2", "2", "2"]],
                overwrite=False,
            )
            lines = history_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(lines[1].split(",")[0], "1509940440000")
        self.assertEqual(lines[2].split(",")[0], "1779694380000")


if __name__ == "__main__":
    unittest.main()
