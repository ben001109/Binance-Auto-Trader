import ast
import unittest

from bat.risk.live_state import LiveState
from bat.tui.app import manual_sell_risk_config


def action_sell_asset_source() -> str:
    path = "src/bat/tui/app.py"
    with open(path, "r", encoding="utf-8") as handle:
        source = handle.read()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "action_sell_asset":
            return ast.get_source_segment(source, node) or ""
    raise AssertionError("action_sell_asset not found")


class ManualSellSafetyTest(unittest.TestCase):
    def test_manual_sell_routes_through_order_gateway(self):
        body = action_sell_asset_source()

        self.assertIn("OrderGateway(", body)
        self.assertIn("OrderIntent(", body)
        self.assertIn("order_gateway.submit(", body)
        self.assertNotIn("broker.sell(", body)

    def test_testnet_manual_sell_uses_testnet_only_canary_guard(self):
        config = manual_sell_risk_config(is_testnet=True)

        self.assertEqual(config.live_state, LiveState.CANARY.value)
        self.assertFalse(config.paper_only)
        self.assertTrue(config.trading_enabled)

    def test_real_manual_sell_uses_default_fail_closed_risk_config(self):
        config = manual_sell_risk_config(is_testnet=False)

        self.assertNotEqual(config.live_state, LiveState.CANARY.value)
        self.assertTrue(config.paper_only)


if __name__ == "__main__":
    unittest.main()
