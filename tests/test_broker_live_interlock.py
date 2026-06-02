import os
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from bat.execution.broker import BinanceBroker


class BrokerLiveInterlockTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._original_interlock = os.environ.get("BAT_ALLOW_REAL_TRADING")
        os.environ.pop("BAT_ALLOW_REAL_TRADING", None)

    def tearDown(self):
        if self._original_interlock is None:
            os.environ.pop("BAT_ALLOW_REAL_TRADING", None)
        else:
            os.environ["BAT_ALLOW_REAL_TRADING"] = self._original_interlock

    async def test_real_buy_without_exact_interlock_returns_none_without_order(self):
        broker = BinanceBroker(is_testnet=False)
        new_order = AsyncMock()

        with patch.object(BinanceBroker, "init_client", AsyncMock()), patch(
            "bat.execution.broker.async_new_order", new_order
        ), patch.object(BinanceBroker, "_adjust_quote_qty", AsyncMock(return_value=20.0)):
            result = await broker.buy(quote_qty=20.0)

        self.assertIsNone(result)
        new_order.assert_not_awaited()

    async def test_real_sell_without_exact_interlock_returns_none_without_order(self):
        broker = BinanceBroker(is_testnet=False)
        new_order = AsyncMock()

        with patch.object(BinanceBroker, "init_client", AsyncMock()), patch(
            "bat.execution.broker.async_new_order", new_order
        ):
            result = await broker.sell(quantity=0.001)

        self.assertIsNone(result)
        new_order.assert_not_awaited()

    async def test_real_buy_with_wrong_interlock_value_returns_none_without_order(self):
        os.environ["BAT_ALLOW_REAL_TRADING"] = "true"
        broker = BinanceBroker(is_testnet=False)
        new_order = AsyncMock()

        with patch.object(BinanceBroker, "init_client", AsyncMock()), patch(
            "bat.execution.broker.async_new_order", new_order
        ), patch.object(BinanceBroker, "_adjust_quote_qty", AsyncMock(return_value=20.0)):
            result = await broker.buy(quote_qty=20.0)

        self.assertIsNone(result)
        new_order.assert_not_awaited()

    async def test_real_buy_with_exact_interlock_reaches_order_submission(self):
        os.environ["BAT_ALLOW_REAL_TRADING"] = "I_UNDERSTAND_REAL_RISK"
        broker = BinanceBroker(is_testnet=False)
        broker.client = object()
        order = SimpleNamespace(order_id="real-buy-1")
        new_order = AsyncMock(return_value=order)

        with patch("bat.execution.broker.async_new_order", new_order), patch.object(
            BinanceBroker, "_adjust_quote_qty", AsyncMock(return_value=20.0)
        ):
            result = await broker.buy(quote_qty=20.0)

        self.assertIs(result, order)
        new_order.assert_awaited_once()

    async def test_testnet_buy_reaches_order_submission_without_interlock(self):
        broker = BinanceBroker(is_testnet=True)
        broker.client = object()
        order = SimpleNamespace(order_id="testnet-buy-1")
        new_order = AsyncMock(return_value=order)

        with patch("bat.execution.broker.async_new_order", new_order), patch.object(
            BinanceBroker, "_adjust_quote_qty", AsyncMock(return_value=20.0)
        ):
            result = await broker.buy(quote_qty=20.0)

        self.assertIs(result, order)
        new_order.assert_awaited_once()

    async def test_testnet_sell_reaches_order_submission_without_interlock(self):
        broker = BinanceBroker(is_testnet=True)
        broker.client = object()
        order = SimpleNamespace(order_id="testnet-sell-1")
        new_order = AsyncMock(return_value=order)

        with patch("bat.execution.broker.async_new_order", new_order):
            result = await broker.sell(quantity=0.001)

        self.assertIs(result, order)
        new_order.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
