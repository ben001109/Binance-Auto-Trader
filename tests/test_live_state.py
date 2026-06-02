import unittest

from bat.research_config import RiskConfig
from bat.risk.live_state import LiveState, LiveStateError, validate_transition


class LiveStateTest(unittest.TestCase):
    def test_default_state_is_paper(self):
        self.assertEqual(RiskConfig().live_state, LiveState.PAPER.value)

    def test_valid_state_transitions(self):
        transitions = [
            (LiveState.OFF, LiveState.PAPER),
            (LiveState.PAPER, LiveState.SHADOW),
            (LiveState.SHADOW, LiveState.CANARY),
            (LiveState.CANARY, LiveState.LIVE_LIMITED),
        ]

        for current, requested in transitions:
            with self.subTest(current=current.value, requested=requested.value):
                self.assertEqual(
                    validate_transition(current.value, requested.value),
                    requested.value,
                )

    def test_invalid_transition_from_paper_to_live_limited_raises(self):
        with self.assertRaises(LiveStateError):
            validate_transition(LiveState.PAPER.value, LiveState.LIVE_LIMITED.value)

    def test_risk_config_rejects_unsupported_live_state(self):
        with self.assertRaises(LiveStateError):
            RiskConfig(live_state="REAL_MONEY")

    def test_emergency_transition_to_off_is_allowed(self):
        states = [
            LiveState.PAPER,
            LiveState.SHADOW,
            LiveState.CANARY,
            LiveState.LIVE_LIMITED,
        ]

        for current in states:
            with self.subTest(current=current.value):
                self.assertEqual(
                    validate_transition(current.value, LiveState.OFF.value),
                    LiveState.OFF.value,
                )

    def test_risk_config_has_canary_defaults(self):
        config = RiskConfig()

        self.assertEqual(config.max_order_equity_fraction, 0.01)
        self.assertEqual(config.max_daily_loss_fraction, 0.005)
        self.assertTrue(config.paper_only)
        self.assertFalse(config.trading_enabled)


if __name__ == "__main__":
    unittest.main()
