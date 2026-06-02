import inspect
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from bat.research_config import RiskConfig
from bat.risk.live_state import LiveState
from bat.services.artifact_security import sha256_file
from bat.tui import app as tui_app
from bat.tui.app import CryptoApp


class FakeWidget:
    def __init__(self, value):
        self.value = value


class TuiLiveReadinessTest(unittest.TestCase):
    def test_save_settings_never_persists_real_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            app = CryptoApp.__new__(CryptoApp)
            app.settings_path = str(Path(tmpdir) / "settings.json")
            widgets = {
                "#select_mode": FakeWidget("REAL"),
                "#select_symbol": FakeWidget("BTCUSDT"),
                "#select_interval": FakeWidget("15m"),
                "#input_poll_interval": FakeWidget("1m"),
            }
            app.query_one = lambda selector, _kind=None: widgets[selector]
            app._get_sim_steps_value = lambda: 120
            app._get_mem_fraction_value = lambda: 0.5
            app._get_online_interval_value = lambda: 1800
            app._get_online_vol_mult_value = lambda: 1.8
            app._get_online_vol_cooldown_value = lambda: 600
            app._get_online_min_trades_value = lambda: 1
            app._get_status_every_value = lambda: 20
            app._get_batch_size_value = lambda: 64
            app._confidence_threshold = lambda: 0.55
            app.wallet_show_all = True
            app.wallet_sort_mode = "amount"

            app._save_settings()

            saved = json.loads(Path(app.settings_path).read_text(encoding="utf-8"))
            self.assertEqual(saved["mode"], "TESTNET")

    def test_real_trading_readiness_requires_typed_arm_canary_and_env_interlock(self):
        risk = RiskConfig(
            live_state=LiveState.CANARY.value,
            paper_only=False,
            trading_enabled=True,
        )

        blocked = tui_app.real_trading_readiness(
            is_testnet=False,
            arm_text="wrong",
            risk_config=risk,
            env={},
            model_ready=True,
        )

        self.assertFalse(blocked.ready)
        self.assertIn("real_not_armed", blocked.reasons)
        self.assertIn("missing_real_interlock", blocked.reasons)

        ready = tui_app.real_trading_readiness(
            is_testnet=False,
            arm_text=tui_app.REAL_TRADING_ARM_PHRASE,
            risk_config=risk,
            env={"BAT_ALLOW_REAL_TRADING": "I_UNDERSTAND_REAL_RISK"},
            model_ready=True,
        )

        self.assertTrue(ready.ready)
        self.assertEqual(ready.reasons, [])

    def test_real_trading_readiness_blocks_non_canary_or_unready_model(self):
        blocked = tui_app.real_trading_readiness(
            is_testnet=False,
            arm_text=tui_app.REAL_TRADING_ARM_PHRASE,
            risk_config=RiskConfig(),
            env={"BAT_ALLOW_REAL_TRADING": "I_UNDERSTAND_REAL_RISK"},
            model_ready=False,
            data_ready=False,
        )

        self.assertFalse(blocked.ready)
        self.assertIn("not_canary", blocked.reasons)
        self.assertIn("paper_only", blocked.reasons)
        self.assertIn("trading_disabled", blocked.reasons)
        self.assertIn("model_not_ready", blocked.reasons)
        self.assertIn("data_not_ready", blocked.reasons)

    def test_live_readiness_status_displays_reasons(self):
        status = tui_app.format_live_readiness_status(
            tui_app.LiveReadiness(ready=False, reasons=["real_not_armed", "missing_real_interlock"])
        )

        self.assertIn("REAL blocked", status)
        self.assertIn("real_not_armed", status)
        self.assertIn("missing_real_interlock", status)

    def test_real_order_paths_check_ui_readiness_before_submit(self):
        auto_source = inspect.getsource(CryptoApp.action_auto_trade)
        sell_source = inspect.getsource(CryptoApp.action_sell_asset)

        self.assertIn("_real_trading_ui_ready", auto_source)
        self.assertIn("_real_trading_ui_ready", sell_source)

    def test_auto_trade_rechecks_readiness_inside_long_running_loop(self):
        auto_source = inspect.getsource(CryptoApp.action_auto_trade)

        self.assertGreaterEqual(auto_source.count("_real_trading_ui_ready"), 2)

    def test_auto_trade_rechecks_readiness_immediately_before_order_submit(self):
        auto_source = inspect.getsource(CryptoApp.action_auto_trade)

        self.assertGreaterEqual(auto_source.count("_real_trading_ui_ready"), 4)

    def test_model_watcher_updates_live_readiness_state(self):
        source = inspect.getsource(CryptoApp._watch_model_ready)

        self.assertIn("self.model_ready", source)
        self.assertIn("_update_live_readiness_status", source)

    def test_current_readiness_checks_runtime_model_artifact(self):
        source = inspect.getsource(CryptoApp._current_live_readiness)

        self.assertIn("runtime_model_ready", source)

    def test_current_readiness_does_not_let_stale_model_ready_bypass_artifact_check(self):
        app = CryptoApp.__new__(CryptoApp)
        app.real_trading_arm_text = tui_app.REAL_TRADING_ARM_PHRASE
        app.model_ready = True
        captured = {}

        def capture_readiness(**kwargs):
            captured.update(kwargs)
            return tui_app.LiveReadiness(ready=False, reasons=[])

        with patch("bat.tui.app.runtime_model_ready", return_value=False), patch(
            "bat.tui.app.load_research_config",
            return_value=SimpleNamespace(
                risk=RiskConfig(
                    live_state=LiveState.CANARY.value,
                    paper_only=False,
                    trading_enabled=True,
                )
            ),
        ), patch("bat.tui.app.real_trading_readiness", side_effect=capture_readiness):
            app._current_live_readiness(is_testnet=False)

        self.assertFalse(captured["model_ready"])

    def test_current_readiness_passes_explicit_data_ready_state(self):
        source = inspect.getsource(CryptoApp._current_live_readiness)

        self.assertIn("data_ready=", source)
        self.assertIn("_runtime_data_ready", source)

    def test_runtime_model_ready_rejects_non_torch_manifest_type(self):
        runs_dir = Path("runs")
        runs_dir.mkdir(exist_ok=True)

        with tempfile.TemporaryDirectory(dir=runs_dir) as tmpdir:
            artifact = Path(tmpdir) / "best_model.pth"
            artifact.write_bytes(b"not a torch state")
            manifest = {
                "artifacts": {
                    artifact.name: {
                        "sha256": sha256_file(artifact),
                        "type": "sklearn_pickle",
                        "runtime_load_allowed": True,
                    }
                },
                "admission": {"passed": True},
            }
            (Path(tmpdir) / "manifest.json").write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )

            self.assertFalse(tui_app.runtime_model_ready(str(artifact)))


if __name__ == "__main__":
    unittest.main()
