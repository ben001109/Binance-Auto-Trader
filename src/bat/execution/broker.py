from bat.config import conf
from decimal import Decimal, ROUND_DOWN

from bat.execution.spot_client import (
    async_account,
    async_exchange_info,
    async_klines,
    async_new_order,
    async_sync_time_offset,
    async_wallet_balances,
    create_spot_client,
    create_wallet_client,
)
from bat.logger import get_logger

class BinanceBroker:
    def __init__(self, is_testnet=None):
        self.client = None
        self.wallet_client = None
        self.symbol = conf.SYMBOL
        self.is_testnet = conf.IS_TESTNET if is_testnet is None else is_testnet
        self.logger = get_logger("bat.broker")
        self._symbol_info = None

    async def init_client(self):
        """初始化連線"""
        if not self.client:
            self.client = create_spot_client(is_testnet=self.is_testnet)
            self.wallet_client = create_wallet_client(is_testnet=self.is_testnet)
            print(f">>> Broker 初始化完成 (Testnet={self.is_testnet})")
            self.logger.info("Broker initialized (Testnet=%s)", self.is_testnet)

    async def get_balance(self, asset='USDT'):
        """查詢餘額"""
        if not self.client: await self.init_client()

        try:
            account = await async_account(self.client)
            for item in account.balances or []:
                if item.asset == asset:
                    self.logger.info("Balance fetched for %s", asset)
                    return float(item.free or 0.0)
            return 0.0
        except Exception as exc:
            if "after shutdown" in str(exc):
                self.logger.warning("Balance fetch skipped during shutdown (%s)", asset)
                return 0.0
            if "recvWindow" in str(exc) or "Timestamp" in str(exc):
                self.logger.warning("Balance fetch retry after time sync for %s", asset)
                await async_sync_time_offset(self.client)
                try:
                    account = await async_account(self.client)
                    for item in account.balances or []:
                        if item.asset == asset:
                            self.logger.info("Balance fetched for %s", asset)
                            return float(item.free or 0.0)
                    return 0.0
                except Exception:
                    pass
            self.logger.exception("Balance fetch failed for %s", asset)
            await self.close()
            raise

    async def get_wallet_overview(self):
        if self.is_testnet:
            self.logger.info("Wallet overview skipped on Testnet")
            return []
        if not self.wallet_client:
            await self.init_client()
        try:
            return await async_wallet_balances(self.wallet_client)
        except Exception:
            self.logger.exception("Wallet overview fetch failed")
            return []

    async def get_klines(self, symbol=None, interval=None, limit=100):
        if not self.client: await self.init_client()
        return await async_klines(
            self.client,
            symbol or self.symbol,
            interval or conf.INTERVAL,
            limit=limit
        )

    async def buy(self, quantity=None, quote_qty=None):
        """
        執行買入
        quantity: 買多少顆 BTC
        quote_qty: 買多少 USDT 的 BTC (例如買 100 U)
        """
        if not self.client: await self.init_client()

        try:
            print(f">>> 執行買入: {self.symbol}")
            if quantity is None and quote_qty is None:
                print("❌ 買入失敗: quantity 或 quote_qty 至少需要提供一個")
                self.logger.warning("Buy aborted: missing quantity/quote_qty")
                return None
            if quote_qty is not None:
                quote_qty = await self._adjust_quote_qty(quote_qty)
            order = await async_new_order(
                self.client,
                symbol=self.symbol,
                side="BUY",
                type="MARKET",
                # 市價單通常用 quoteOrderQty (我想買 100 U) 或 quantity (我想買 0.01 BTC)
                quantity=quantity,
                quote_order_qty=quote_qty,
            )
            print(f"✅ 買入成功: {order.order_id}")
            self.logger.info("Buy success: %s", order.order_id)
            return order
        except Exception as e:
            print(f"❌ 買入失敗: {e}")
            self.logger.exception("Buy failed")
            await self.close()
            return None

    async def sell(self, quantity):
        """執行賣出 (賣出多少顆 BTC)"""
        if not self.client: await self.init_client()

        try:
            print(f">>> 執行賣出: {self.symbol}")
            order = await async_new_order(
                self.client,
                symbol=self.symbol,
                side="SELL",
                type="MARKET",
                quantity=quantity
            )
            print(f"✅ 賣出成功: {order.order_id}")
            self.logger.info("Sell success: %s", order.order_id)
            return order
        except Exception as e:
            print(f"❌ 賣出失敗: {e}")
            self.logger.exception("Sell failed")
            await self.close()
            return None

    async def close(self):
        if self.client:
            self.client = None

    async def _load_symbol_info(self):
        if self._symbol_info is None:
            info = await async_exchange_info(self.client, self.symbol)
            symbols = info.get("symbols", []) if isinstance(info, dict) else []
            self._symbol_info = symbols[0] if symbols else {}
        return self._symbol_info

    def _as_dict(self, obj):
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
        return obj

    def _get_filter(self, symbol_info, filter_type):
        filters = symbol_info.get("filters", []) if isinstance(symbol_info, dict) else []
        for item in filters:
            data = self._as_dict(item)
            if data.get("filterType") == filter_type:
                return data
        return {}

    def _round_step(self, value, step):
        step_dec = Decimal(str(step))
        value_dec = Decimal(str(value))
        return float(value_dec.quantize(step_dec, rounding=ROUND_DOWN))

    async def _adjust_quote_qty(self, quote_qty):
        info = await self._load_symbol_info()
        market_filter = self._get_filter(info, "MARKET_LOT_SIZE")
        step_size = market_filter.get("stepSize")
        if step_size:
            adjusted = self._round_step(quote_qty, step_size)
            if adjusted > 0:
                return adjusted
        quote_precision = info.get("quotePrecision")
        if quote_precision is not None:
            precision = int(quote_precision)
            scale = Decimal("1e-{0}".format(precision))
            return float(Decimal(str(quote_qty)).quantize(scale, rounding=ROUND_DOWN))
        return float(round(quote_qty, 8))
