import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bat.training import set_stop_training, should_stop_training
from bat.tui.app import CryptoApp


class TuiHistoryDownloadTest(unittest.IsolatedAsyncioTestCase):
    async def asyncTearDown(self):
        set_stop_training(False)

    async def test_download_resets_stop_flag_before_fetching(self):
        observed = {}

        async def fake_historical_klines(*args, **kwargs):
            observed["stopped"] = should_stop_training()
            return []

        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                app = CryptoApp.__new__(CryptoApp)
                app._read_inputs = lambda: ("BTCUSDT", "1m", "1m", 120, True)
                app.log_train = lambda *_args, **_kwargs: None
                app.log_train_error = lambda *_args, **_kwargs: None
                app._onboard_date_str = lambda _info: "1970-01-01 00:00:00"
                app._parse_date_ms = lambda _value: 0
                app._interval_ms_for_klines = lambda _interval: 60_000
                app._progress_bar = lambda _percent: ""
                app._update_train_status_threadsafe = lambda *_args, **_kwargs: None

                set_stop_training(True)
                with patch("bat.tui.app.create_spot_client", return_value=object()), patch(
                    "bat.tui.app.async_exchange_info",
                    return_value={},
                ), patch(
                    "bat.tui.app.async_historical_klines",
                    side_effect=fake_historical_klines,
                ):
                    await app.action_download_history()
            finally:
                os.chdir(cwd)

        self.assertFalse(observed["stopped"])

    async def test_existing_history_without_metadata_is_backed_up_before_download(self):
        observed = {"download_called": False}

        async def fake_historical_klines(*args, **kwargs):
            observed["download_called"] = True
            return []

        with tempfile.TemporaryDirectory() as tmpdir:
            cwd = os.getcwd()
            os.chdir(tmpdir)
            try:
                data_dir = Path("data")
                data_dir.mkdir()
                (data_dir / "history.csv").write_text(
                    "timestamp,open\n999999999999999,1\n",
                    encoding="utf-8",
                )

                app = CryptoApp.__new__(CryptoApp)
                app._read_inputs = lambda: ("BTCUSDT", "1m", "1m", 120, True)
                app.log_train = lambda *_args, **_kwargs: None
                app.log_train_error = lambda *_args, **_kwargs: None
                app._onboard_date_str = lambda _info: "1970-01-01 00:00:01"
                app._parse_date_ms = lambda _value: 1000
                app._interval_ms_for_klines = lambda _interval: 60_000
                app._progress_bar = lambda _percent: ""
                app._update_train_status_threadsafe = lambda *_args, **_kwargs: None

                with patch("bat.tui.app.create_spot_client", return_value=object()), patch(
                    "bat.tui.app.async_exchange_info",
                    return_value={},
                ), patch("bat.tui.app.time.time", return_value=123), patch(
                    "bat.tui.app.async_historical_klines",
                    side_effect=fake_historical_klines,
                ):
                    await app.action_download_history()
                backup_exists = (data_dir / "history.csv.bak.123").exists()
            finally:
                os.chdir(cwd)

        self.assertTrue(observed["download_called"])
        self.assertTrue(backup_exists)


if __name__ == "__main__":
    unittest.main()
