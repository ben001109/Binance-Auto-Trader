import unittest
from math import inf, nan

from bat.research_config import RiskConfig
from bat.risk.live_guard import AccountRiskSnapshot, LiveGuard, OrderIntent
from bat.risk.live_state import LiveState


class LiveGuardTest(unittest.TestCase):
    def setUp(self):
        self.account = AccountRiskSnapshot(
            equity_usdt=10_000.0,
            daily_pnl_fraction=0.0,
            consecutive_losses=0,
        )
        self.intent = OrderIntent(
            symbol="BTCUSDT",
            side="BUY",
            quote_qty=50.0,
            quantity=None,
            confidence=0.80,
            data_age_seconds=60.0,
        )

    def guard(self, **overrides):
        values = {
            "live_state": LiveState.CANARY.value,
            "paper_only": False,
            "trading_enabled": True,
        }
        values.update(overrides)
        config = RiskConfig(**values)
        return LiveGuard(config)

    def decision(self, *, guard=None, intent=None, account=None):
        return (guard or self.guard()).evaluate_order_request(
            intent or self.intent,
            account or self.account,
        )

    def test_blocks_real_orders_when_state_is_not_canary(self):
        for state in [LiveState.OFF, LiveState.PAPER, LiveState.SHADOW]:
            with self.subTest(state=state.value):
                decision = self.decision(
                    guard=self.guard(live_state=state.value),
                )

                self.assertFalse(decision.allowed)
                self.assertEqual(decision.state, state.value)
                self.assertEqual(decision.reason, "state_not_live")
                self.assertIsNone(decision.adjusted_quote_qty)

    def test_blocks_when_paper_only_is_enabled(self):
        decision = self.decision(guard=self.guard(paper_only=True))

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "paper_only")

    def test_blocks_when_trading_is_disabled(self):
        decision = self.decision(guard=self.guard(trading_enabled=False))

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "trading_disabled")

    def test_blocks_symbol_other_than_btcusdt(self):
        intent = OrderIntent(
            symbol="ETHUSDT",
            side="BUY",
            quote_qty=50.0,
            quantity=None,
            confidence=0.80,
            data_age_seconds=60.0,
        )

        decision = self.decision(intent=intent)

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "symbol_not_allowed")

    def test_blocks_when_confidence_is_below_threshold(self):
        intent = OrderIntent(
            symbol="BTCUSDT",
            side="BUY",
            quote_qty=50.0,
            quantity=None,
            confidence=0.69,
            data_age_seconds=60.0,
        )

        decision = self.decision(intent=intent)

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "confidence_below_threshold")

    def test_caps_quote_amount_to_one_percent_equity_in_canary(self):
        intent = OrderIntent(
            symbol="BTCUSDT",
            side="BUY",
            quote_qty=500.0,
            quantity=None,
            confidence=0.80,
            data_age_seconds=60.0,
        )

        decision = self.decision(intent=intent)

        self.assertTrue(decision.allowed)
        self.assertEqual(decision.reason, "allowed")
        self.assertEqual(decision.adjusted_quote_qty, 100.0)

    def test_blocks_when_daily_loss_limit_is_reached(self):
        account = AccountRiskSnapshot(
            equity_usdt=10_000.0,
            daily_pnl_fraction=-0.005,
            consecutive_losses=0,
        )

        decision = self.decision(account=account)

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "daily_loss_limit")

    def test_blocks_when_consecutive_loss_limit_is_reached(self):
        account = AccountRiskSnapshot(
            equity_usdt=10_000.0,
            daily_pnl_fraction=0.0,
            consecutive_losses=2,
        )

        decision = self.decision(account=account)

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "consecutive_loss_limit")

    def test_blocks_when_consecutive_losses_is_not_finite_or_negative(self):
        invalid_values = [nan, inf, -1]

        for consecutive_losses in invalid_values:
            with self.subTest(consecutive_losses=consecutive_losses):
                account = AccountRiskSnapshot(
                    equity_usdt=10_000.0,
                    daily_pnl_fraction=0.0,
                    consecutive_losses=consecutive_losses,
                )

                decision = self.decision(account=account)

                self.assertFalse(decision.allowed)
                self.assertEqual(decision.reason, "invalid_consecutive_losses")

    def test_blocks_when_data_is_stale(self):
        intent = OrderIntent(
            symbol="BTCUSDT",
            side="BUY",
            quote_qty=50.0,
            quantity=None,
            confidence=0.80,
            data_age_seconds=1201.0,
        )

        decision = self.decision(intent=intent)

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "stale_data")

    def test_allows_canary_real_trading_when_checks_pass(self):
        decision = self.decision()

        self.assertTrue(decision.allowed)
        self.assertEqual(decision.state, LiveState.CANARY.value)
        self.assertEqual(decision.reason, "allowed")
        self.assertEqual(decision.adjusted_quote_qty, 50.0)

    def test_allows_sell_without_quote_quantity_for_gateway_quantity_handling(self):
        intent = OrderIntent(
            symbol="BTCUSDT",
            side="SELL",
            quote_qty=None,
            quantity=0.01,
            confidence=0.80,
            data_age_seconds=60.0,
        )

        decision = self.decision(intent=intent)

        self.assertTrue(decision.allowed)
        self.assertEqual(decision.reason, "allowed")
        self.assertIsNone(decision.adjusted_quote_qty)
        self.assertEqual(decision.adjusted_quantity, 0.01)

    def test_blocks_sell_when_quantity_is_not_positive_and_finite(self):
        invalid_quantities = [None, 0.0, -0.01, nan, inf]

        for quantity in invalid_quantities:
            with self.subTest(quantity=quantity):
                intent = OrderIntent(
                    symbol="BTCUSDT",
                    side="SELL",
                    quote_qty=None,
                    quantity=quantity,
                    confidence=0.80,
                    data_age_seconds=60.0,
                )

                decision = self.decision(intent=intent)

                self.assertFalse(decision.allowed)
                self.assertEqual(decision.reason, "invalid_order_size")

    def test_blocks_buy_when_quote_quantity_is_not_finite(self):
        intent = OrderIntent(
            symbol="BTCUSDT",
            side="BUY",
            quote_qty=nan,
            quantity=None,
            confidence=0.80,
            data_age_seconds=60.0,
        )

        decision = self.decision(intent=intent)

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "invalid_order_size")

    def test_blocks_when_confidence_is_not_finite(self):
        intent = OrderIntent(
            symbol="BTCUSDT",
            side="BUY",
            quote_qty=50.0,
            quantity=None,
            confidence=nan,
            data_age_seconds=60.0,
        )

        decision = self.decision(intent=intent)

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "invalid_confidence")

    def test_blocks_when_data_age_is_not_finite(self):
        intent = OrderIntent(
            symbol="BTCUSDT",
            side="BUY",
            quote_qty=50.0,
            quantity=None,
            confidence=0.80,
            data_age_seconds=nan,
        )

        decision = self.decision(intent=intent)

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "invalid_data_age")

    def test_blocks_when_equity_is_not_finite(self):
        account = AccountRiskSnapshot(
            equity_usdt=nan,
            daily_pnl_fraction=0.0,
            consecutive_losses=0,
        )

        decision = self.decision(account=account)

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "invalid_equity")

    def test_blocks_when_daily_pnl_is_not_finite(self):
        account = AccountRiskSnapshot(
            equity_usdt=10_000.0,
            daily_pnl_fraction=nan,
            consecutive_losses=0,
        )

        decision = self.decision(account=account)

        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "invalid_daily_pnl")


if __name__ == "__main__":
    unittest.main()
