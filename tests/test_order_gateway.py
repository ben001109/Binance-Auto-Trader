import unittest

from bat.execution.order_gateway import OrderGateway
from bat.research_config import RiskConfig
from bat.risk.live_guard import AccountRiskSnapshot, GuardDecision, LiveGuard, OrderIntent
from bat.risk.live_state import LiveState


class FakeBroker:
    def __init__(self, *, symbol="BTCUSDT", buy_order=None, sell_order=None):
        self.symbol = symbol
        self.buy_calls = []
        self.sell_calls = []
        self.buy_order = {"order_id": "buy-123", "status": "FILLED"} if buy_order == "default" or buy_order is None else buy_order
        self.sell_order = {"orderId": "sell-123", "status": "FILLED"} if sell_order == "default" or sell_order is None else sell_order

    async def buy(self, *, quote_qty, client_order_id=None):
        self.buy_calls.append(quote_qty)
        return self.buy_order

    async def sell(self, *, quantity, client_order_id=None):
        self.sell_calls.append(quantity)
        return self.sell_order


class RaisingBroker(FakeBroker):
    async def buy(self, *, quote_qty, client_order_id=None):
        self.buy_calls.append(quote_qty)
        raise RuntimeError("broker failed signature=abcdef")

    async def sell(self, *, quantity, client_order_id=None):
        self.sell_calls.append(quantity)
        raise RuntimeError("broker failed signature=abcdef")


class FakeGuard:
    def __init__(self, decision):
        self.decision = decision

    def evaluate_order_request(self, intent, account):
        return self.decision


class FakeLedger:
    def __init__(self):
        self.events = []

    def append(self, event_type, payload):
        self.events.append((event_type, payload))
        return {"event_type": event_type, "payload": payload}


class OrderGatewayTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.account = AccountRiskSnapshot(
            equity_usdt=10_000.0,
            daily_pnl_fraction=0.0,
            consecutive_losses=0,
        )
        self.buy_intent = OrderIntent(
            symbol="BTCUSDT",
            side="BUY",
            quote_qty=500.0,
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
        return LiveGuard(RiskConfig(**values))

    async def test_blocked_guard_decision_never_calls_broker(self):
        broker = FakeBroker()
        gateway = OrderGateway(broker, self.guard(trading_enabled=False))

        result = await gateway.submit(self.buy_intent, self.account)

        self.assertFalse(result.sent)
        self.assertEqual(result.blocked_reason, "trading_disabled")
        self.assertIsNone(result.order_id)
        self.assertIsNone(result.order)
        self.assertIs(result.intent, self.buy_intent)
        self.assertFalse(result.guard_decision.allowed)
        self.assertEqual(broker.buy_calls, [])
        self.assertEqual(broker.sell_calls, [])

    async def test_shadow_records_intended_order_but_never_calls_broker(self):
        broker = FakeBroker()
        gateway = OrderGateway(
            broker,
            self.guard(live_state=LiveState.SHADOW.value),
        )

        result = await gateway.submit(self.buy_intent, self.account)

        self.assertFalse(result.sent)
        self.assertEqual(result.blocked_reason, "shadow_mode")
        self.assertIsNone(result.order_id)
        self.assertIsNone(result.order)
        self.assertIs(result.intent, self.buy_intent)
        self.assertEqual(result.guard_decision.state, LiveState.SHADOW.value)
        self.assertEqual(broker.buy_calls, [])
        self.assertEqual(broker.sell_calls, [])

    async def test_canary_buy_calls_broker_with_adjusted_quote_qty(self):
        broker = FakeBroker()
        gateway = OrderGateway(broker, self.guard())

        result = await gateway.submit(self.buy_intent, self.account)

        self.assertTrue(result.sent)
        self.assertIsNone(result.blocked_reason)
        self.assertEqual(result.order_id, "buy-123")
        self.assertEqual(result.order, {"order_id": "buy-123", "status": "FILLED"})
        self.assertTrue(result.guard_decision.allowed)
        self.assertEqual(result.guard_decision.adjusted_quote_qty, 100.0)
        self.assertEqual(broker.buy_calls, [100.0])
        self.assertEqual(broker.sell_calls, [])

    async def test_canary_sell_calls_broker_with_validated_quantity(self):
        broker = FakeBroker()
        gateway = OrderGateway(
            broker,
            FakeGuard(
                GuardDecision(
                    allowed=True,
                    state=LiveState.CANARY.value,
                    reason="allowed",
                    adjusted_quote_qty=None,
                    adjusted_quantity=0.012,
                )
            ),
        )
        intent = OrderIntent(
            symbol="BTCUSDT",
            side="SELL",
            quote_qty=None,
            quantity=0.025,
            confidence=0.80,
            data_age_seconds=60.0,
        )

        result = await gateway.submit(intent, self.account)

        self.assertTrue(result.sent)
        self.assertIsNone(result.blocked_reason)
        self.assertEqual(result.order_id, "sell-123")
        self.assertEqual(result.order, {"orderId": "sell-123", "status": "FILLED"})
        self.assertEqual(broker.buy_calls, [])
        self.assertEqual(broker.sell_calls, [0.012])

    async def test_broker_symbol_mismatch_fails_closed_without_buy_call(self):
        broker = FakeBroker(symbol="ETHUSDT")
        gateway = OrderGateway(broker, self.guard())

        result = await gateway.submit(self.buy_intent, self.account)

        self.assertFalse(result.sent)
        self.assertEqual(result.blocked_reason, "broker_symbol_mismatch")
        self.assertIsNone(result.order_id)
        self.assertIsNone(result.order)
        self.assertEqual(broker.buy_calls, [])
        self.assertEqual(broker.sell_calls, [])

    async def test_broker_symbol_mismatch_fails_closed_without_sell_call(self):
        broker = FakeBroker(symbol="ETHUSDT")
        gateway = OrderGateway(broker, self.guard())
        intent = OrderIntent(
            symbol="BTCUSDT",
            side="SELL",
            quote_qty=None,
            quantity=0.025,
            confidence=0.80,
            data_age_seconds=60.0,
        )

        result = await gateway.submit(intent, self.account)

        self.assertFalse(result.sent)
        self.assertEqual(result.blocked_reason, "broker_symbol_mismatch")
        self.assertIsNone(result.order_id)
        self.assertIsNone(result.order)
        self.assertEqual(broker.buy_calls, [])
        self.assertEqual(broker.sell_calls, [])

    async def test_buy_returning_none_is_not_reported_as_sent(self):
        broker = FakeBroker(buy_order=False)
        broker.buy_order = None
        gateway = OrderGateway(broker, self.guard())

        result = await gateway.submit(self.buy_intent, self.account)

        self.assertFalse(result.sent)
        self.assertEqual(result.blocked_reason, "execution_unknown")
        self.assertIsNone(result.order_id)
        self.assertIsNone(result.order)
        self.assertEqual(broker.buy_calls, [100.0])
        self.assertEqual(broker.sell_calls, [])

    async def test_audit_ledger_records_blocked_without_raw_intent_amounts(self):
        broker = FakeBroker()
        ledger = FakeLedger()
        gateway = OrderGateway(broker, self.guard(trading_enabled=False), audit_ledger=ledger)

        await gateway.submit(self.buy_intent, self.account)

        self.assertEqual(len(ledger.events), 1)
        event_type, payload = ledger.events[0]
        self.assertEqual(event_type, "order_blocked")
        self.assertEqual(payload["symbol"], "BTCUSDT")
        self.assertEqual(payload["side"], "BUY")
        self.assertEqual(payload["state"], LiveState.CANARY.value)
        self.assertEqual(payload["reason"], "trading_disabled")
        self.assertFalse(payload["sent"])
        self.assertNotIn("intent", payload)
        self.assertNotIn("quote_qty", payload)
        self.assertNotIn("quantity", payload)
        self.assertNotIn("confidence", payload)

    async def test_audit_ledger_records_sent_order_without_raw_intent_amounts(self):
        broker = FakeBroker()
        ledger = FakeLedger()
        gateway = OrderGateway(broker, self.guard(), audit_ledger=ledger)

        await gateway.submit(self.buy_intent, self.account)

        self.assertEqual(len(ledger.events), 1)
        event_type, payload = ledger.events[0]
        self.assertEqual(event_type, "order_sent")
        self.assertEqual(payload["symbol"], "BTCUSDT")
        self.assertEqual(payload["side"], "BUY")
        self.assertEqual(payload["state"], LiveState.CANARY.value)
        self.assertEqual(payload["reason"], "allowed")
        self.assertTrue(payload["sent"])
        self.assertEqual(payload["order_id"], "buy-123")
        self.assertTrue(payload["client_order_id"].startswith("bat-BTCUSDT-BUY-"))
        self.assertNotIn("intent", payload)
        self.assertNotIn("quote_qty", payload)
        self.assertNotIn("quantity", payload)
        self.assertNotIn("confidence", payload)

    async def test_sell_returning_none_is_not_reported_as_sent(self):
        broker = FakeBroker(sell_order=False)
        broker.sell_order = None
        gateway = OrderGateway(broker, self.guard())
        intent = OrderIntent(
            symbol="BTCUSDT",
            side="SELL",
            quote_qty=None,
            quantity=0.025,
            confidence=0.80,
            data_age_seconds=60.0,
        )

        result = await gateway.submit(intent, self.account)

        self.assertFalse(result.sent)
        self.assertEqual(result.blocked_reason, "execution_unknown")
        self.assertIsNone(result.order_id)
        self.assertIsNone(result.order)
        self.assertEqual(broker.buy_calls, [])
        self.assertEqual(broker.sell_calls, [0.025])

    async def test_buy_exception_is_fail_closed_and_audited_without_raw_amounts(self):
        broker = RaisingBroker()
        ledger = FakeLedger()
        gateway = OrderGateway(broker, self.guard(), audit_ledger=ledger)

        result = await gateway.submit(self.buy_intent, self.account)
        retry = await gateway.submit(self.buy_intent, self.account)

        self.assertFalse(result.sent)
        self.assertEqual(result.blocked_reason, "execution_unknown")
        self.assertIsNone(result.order_id)
        self.assertIsNone(result.order)
        self.assertEqual(broker.buy_calls, [100.0])
        self.assertEqual(retry.blocked_reason, "in_flight_order")
        event_type, payload = ledger.events[0]
        self.assertEqual(event_type, "broker_exception")
        self.assertEqual(payload["symbol"], "BTCUSDT")
        self.assertEqual(payload["side"], "BUY")
        self.assertEqual(payload["reason"], "execution_unknown")
        self.assertFalse(payload["sent"])
        self.assertNotIn("quote_qty", payload)
        self.assertNotIn("quantity", payload)
        self.assertNotIn("500.0", str(payload))

    async def test_sell_exception_is_fail_closed_and_audited_without_raw_amounts(self):
        broker = RaisingBroker()
        ledger = FakeLedger()
        gateway = OrderGateway(
            broker,
            FakeGuard(
                GuardDecision(
                    allowed=True,
                    state=LiveState.CANARY.value,
                    reason="allowed",
                    adjusted_quote_qty=None,
                    adjusted_quantity=0.012,
                )
            ),
            audit_ledger=ledger,
        )
        intent = OrderIntent(
            symbol="BTCUSDT",
            side="SELL",
            quote_qty=None,
            quantity=0.025,
            confidence=0.80,
            data_age_seconds=60.0,
        )

        result = await gateway.submit(intent, self.account)

        self.assertFalse(result.sent)
        self.assertEqual(result.blocked_reason, "execution_unknown")
        self.assertEqual(broker.sell_calls, [0.012])
        event_type, payload = ledger.events[0]
        self.assertEqual(event_type, "broker_exception")
        self.assertEqual(payload["side"], "SELL")
        self.assertNotIn("quantity", payload)
        self.assertNotIn("0.012", str(payload))
        self.assertNotIn("0.025", str(payload))


if __name__ == "__main__":
    unittest.main()
