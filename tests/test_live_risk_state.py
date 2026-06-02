import inspect
import math
import tempfile
import unittest
from pathlib import Path

from bat.config import Config
from bat.research_config import RiskConfig
from bat.risk.live_guard import AccountRiskSnapshot, LiveGuard, OrderIntent
from bat.risk.live_state import LiveState
from bat.risk.risk_state import latest_kline_data_age_seconds, risk_state_from_trade_ledger
from bat.tui.app import CryptoApp


class LiveRiskStateTest(unittest.TestCase):
    def test_latest_kline_data_age_seconds_uses_latest_close_time(self):
        klines = [
            [0, "1", "2", "0.5", "1.5", "10", 1_000],
            [0, "1", "2", "0.5", "1.5", "10", 2_500],
        ]

        age = latest_kline_data_age_seconds(klines, now_ms=10_000)

        self.assertEqual(age, 7.5)

    def test_latest_kline_data_age_seconds_returns_nan_for_empty_or_malformed_klines(self):
        malformed_cases = [
            [],
            [[0, 1, 2]],
            [[0, 1, 2, 3, 4, 5, "not-a-time"]],
            None,
        ]

        for klines in malformed_cases:
            with self.subTest(klines=klines):
                self.assertTrue(math.isnan(latest_kline_data_age_seconds(klines, now_ms=10_000)))

    def test_latest_kline_data_age_seconds_returns_nan_for_future_close_time(self):
        klines = [[0, "1", "2", "0.5", "1.5", "10", 20_000]]

        age = latest_kline_data_age_seconds(klines, now_ms=10_000)

        self.assertTrue(math.isnan(age))

    def test_default_live_readiness_symbol_is_btcusdt(self):
        self.assertEqual(Config.SYMBOL, "BTCUSDT")

    def test_real_missing_ledger_returns_unavailable_nan_and_live_guard_blocks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = risk_state_from_trade_ledger(Path(temp_dir) / "missing.csv", is_testnet=False)

        self.assertFalse(snapshot.available)
        self.assertTrue(math.isnan(snapshot.daily_pnl_fraction))
        self.assertEqual(snapshot.consecutive_losses, 0)

        guard = LiveGuard(
            RiskConfig(
                live_state=LiveState.CANARY.value,
                paper_only=False,
                trading_enabled=True,
            )
        )
        decision = guard.evaluate_order_request(
            OrderIntent(
                symbol="BTCUSDT",
                side="BUY",
                quote_qty=50.0,
                quantity=None,
                confidence=0.80,
                data_age_seconds=60.0,
            ),
            AccountRiskSnapshot(
                equity_usdt=10_000.0,
                daily_pnl_fraction=snapshot.daily_pnl_fraction,
                consecutive_losses=snapshot.consecutive_losses,
            ),
        )

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "invalid_daily_pnl")

    def test_testnet_missing_ledger_returns_available_zero_risk_state(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            snapshot = risk_state_from_trade_ledger(Path(temp_dir) / "missing.csv", is_testnet=True)

        self.assertTrue(snapshot.available)
        self.assertEqual(snapshot.daily_pnl_fraction, 0.0)
        self.assertEqual(snapshot.consecutive_losses, 0)

    def test_action_auto_trade_does_not_use_dummy_risk_or_data_age_in_order_intents(self):
        source = inspect.getsource(CryptoApp.action_auto_trade)

        self.assertNotIn("daily_pnl_fraction=0.0", source)
        self.assertNotIn("data_age_seconds=0.0", source)
        self.assertIn("risk_state_from_trade_ledger", source)
        self.assertIn("latest_kline_data_age_seconds", source)


if __name__ == "__main__":
    unittest.main()
