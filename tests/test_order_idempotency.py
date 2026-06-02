import unittest

from bat.execution.order_gateway import OrderGateway
from bat.execution.spot_client import async_new_order
from bat.research_config import RiskConfig
from bat.risk.live_guard import AccountRiskSnapshot, LiveGuard, OrderIntent
from bat.risk.live_state import LiveState


_DEFAULT_ORDER = object()


class FakeBroker:
    def __init__(self, *, order=_DEFAULT_ORDER):
        self.symbol = "BTCUSDT"
        self.order = {"order_id": "order-1"} if order is _DEFAULT_ORDER else order
        self.buy_calls = []
        self.sell_calls = []

    async def buy(self, *, quote_qty, client_order_id=None):
        self.buy_calls.append(
            {"quote_qty": quote_qty, "client_order_id": client_order_id}
        )
        return self.order

    async def sell(self, *, quantity, client_order_id=None):
        self.sell_calls.append(
            {"quantity": quantity, "client_order_id": client_order_id}
        )
        return self.order


class OrderIdempotencyTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.account = AccountRiskSnapshot(
            equity_usdt=10_000.0,
            daily_pnl_fraction=0.0,
            consecutive_losses=0,
        )
        self.intent = OrderIntent(
            symbol="BTCUSDT",
            side="BUY",
            quote_qty=500.0,
            quantity=None,
            confidence=0.80,
            data_age_seconds=60.0,
        )

    def guard(self):
        return LiveGuard(
            RiskConfig(
                live_state=LiveState.CANARY.value,
                paper_only=False,
                trading_enabled=True,
            )
        )

    async def test_gateway_uses_deterministic_client_order_id_for_same_intent(self):
        broker = FakeBroker()
        gateway = OrderGateway(broker, self.guard())

        first = await gateway.submit(self.intent, self.account)
        second = await gateway.submit(self.intent, self.account)

        self.assertTrue(first.sent)
        self.assertTrue(second.sent)
        first_client_id = broker.buy_calls[0]["client_order_id"]
        second_client_id = broker.buy_calls[1]["client_order_id"]
        self.assertEqual(first_client_id, second_client_id)
        self.assertLessEqual(len(first_client_id), 36)
        self.assertRegex(first_client_id, r"^bat-BTCUSDT-BUY-[A-Za-z0-9_-]+$")

    async def test_gateway_keeps_unknown_order_in_flight_and_blocks_repeat(self):
        broker = FakeBroker(order=None)
        gateway = OrderGateway(broker, self.guard())

        first = await gateway.submit(self.intent, self.account)
        second = await gateway.submit(self.intent, self.account)

        self.assertFalse(first.sent)
        self.assertEqual(first.blocked_reason, "execution_unknown")
        self.assertFalse(second.sent)
        self.assertEqual(second.blocked_reason, "in_flight_order")
        self.assertEqual(len(broker.buy_calls), 1)

    async def test_gateway_tracks_in_flight_by_symbol_and_side(self):
        broker = FakeBroker(order=None)
        gateway = OrderGateway(broker, self.guard())
        sell_intent = OrderIntent(
            symbol="BTCUSDT",
            side="SELL",
            quote_qty=None,
            quantity=0.02,
            confidence=0.80,
            data_age_seconds=60.0,
        )

        buy_result = await gateway.submit(self.intent, self.account)
        sell_result = await gateway.submit(sell_intent, self.account)

        self.assertEqual(buy_result.blocked_reason, "execution_unknown")
        self.assertEqual(sell_result.blocked_reason, "execution_unknown")
        self.assertEqual(len(broker.buy_calls), 1)
        self.assertEqual(len(broker.sell_calls), 1)

    def test_client_order_id_is_ascii_safe_and_never_exceeds_binance_limit(self):
        intent = OrderIntent(
            symbol="VERY-LONG-SYMBOL/USDT",
            side="BUY-LONG",
            quote_qty=500.0,
            quantity=None,
            confidence=0.80,
            data_age_seconds=60.0,
        )

        client_order_id = OrderGateway._client_order_id(intent)

        self.assertLessEqual(len(client_order_id), 36)
        self.assertRegex(client_order_id, r"^[A-Za-z0-9_-]+$")

    def test_client_order_id_ignores_volatile_confidence_and_data_age(self):
        retry_intent = OrderIntent(
            symbol=self.intent.symbol,
            side=self.intent.side,
            quote_qty=self.intent.quote_qty,
            quantity=self.intent.quantity,
            confidence=0.99,
            data_age_seconds=900.0,
        )

        first_client_order_id = OrderGateway._client_order_id(self.intent)
        retry_client_order_id = OrderGateway._client_order_id(retry_intent)

        self.assertEqual(first_client_order_id, retry_client_order_id)


class SpotClientOrderIdTest(unittest.IsolatedAsyncioTestCase):
    def client(self):
        class Response:
            def data(self):
                return {"order_id": "order-1"}

        class RestAPI:
            def __init__(self):
                self.payload = None

            def new_order(self, **payload):
                self.payload = payload
                return Response()

        class Client:
            def __init__(self):
                self.rest_api = RestAPI()

        return Client()

    async def test_async_new_order_passes_new_client_order_id_to_sdk_payload(self):
        client = self.client()

        result = await async_new_order(
            client,
            symbol="BTCUSDT",
            side="BUY",
            type="MARKET",
            quote_order_qty=100.0,
            new_client_order_id="bat-BTCUSDT-BUY-abc123",
        )

        self.assertEqual(result, {"order_id": "order-1"})
        self.assertEqual(
            client.rest_api.payload["new_client_order_id"],
            "bat-BTCUSDT-BUY-abc123",
        )
        self.assertNotIn("newClientOrderId", client.rest_api.payload)

    async def test_async_new_order_omits_empty_client_order_id(self):
        client = self.client()

        await async_new_order(
            client,
            symbol="BTCUSDT",
            side="BUY",
            type="MARKET",
            quote_order_qty=100.0,
            new_client_order_id=None,
        )

        self.assertNotIn("new_client_order_id", client.rest_api.payload)


if __name__ == "__main__":
    unittest.main()
