import asyncio
import inspect
import time
from datetime import datetime, timezone

import pandas as pd
from binance_common.configuration import ConfigurationRestAPI
from binance_common import utils as binance_utils
from binance_sdk_spot import Spot
from binance_sdk_spot.rest_api.models.enums import (
    KlinesIntervalEnum,
    NewOrderSideEnum,
    NewOrderTypeEnum,
)
from binance_sdk_wallet import Wallet

from bat.config import conf

TESTNET_BASE_URL = "https://testnet.binance.vision"
KLINES_LIMIT = 1000
_TIME_OFFSET_MS = 0


def _normalize_klines(klines):
    normalized = []
    for item in klines or []:
        if hasattr(item, "actual_instance"):
            normalized.append(item.actual_instance)
        else:
            normalized.append(item)
    return normalized


def create_spot_client(is_testnet=None, api_key=None, api_secret=None):
    use_testnet = conf.IS_TESTNET if is_testnet is None else is_testnet
    if api_key is None or api_secret is None:
        conf.validate_api_config(use_testnet)
        api_key, api_secret = conf.get_credentials(use_testnet)
    base_path = TESTNET_BASE_URL if use_testnet else None
    config = ConfigurationRestAPI(api_key=api_key, api_secret=api_secret, base_path=base_path)
    client = Spot(config_rest_api=config)
    _sync_time_offset(client)
    return client


def create_wallet_client(is_testnet=None, api_key=None, api_secret=None):
    use_testnet = conf.IS_TESTNET if is_testnet is None else is_testnet
    if api_key is None or api_secret is None:
        conf.validate_api_config(use_testnet)
        api_key, api_secret = conf.get_credentials(use_testnet)
    base_path = TESTNET_BASE_URL if use_testnet else None
    config = ConfigurationRestAPI(api_key=api_key, api_secret=api_secret, base_path=base_path)
    client = Wallet(config_rest_api=config)
    return client


def _interval_to_enum(interval: str) -> KlinesIntervalEnum:
    return KlinesIntervalEnum(interval)


def _interval_to_ms(interval: str) -> int:
    unit = interval[-1]
    value = int(interval[:-1])
    if unit == "m":
        return value * 60 * 1000
    if unit == "h":
        return value * 60 * 60 * 1000
    if unit == "d":
        return value * 24 * 60 * 60 * 1000
    if unit == "w":
        return value * 7 * 24 * 60 * 60 * 1000
    if unit == "M":
        return value * 30 * 24 * 60 * 60 * 1000
    raise ValueError(f"Unsupported interval: {interval}")


def _parse_date(value: str) -> int:
    if value.lower() == "now":
        return int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    dt = pd.to_datetime(value, utc=True)
    return int(dt.timestamp() * 1000)


async def async_klines(client: Spot, symbol: str, interval: str, **kwargs):
    interval_enum = _interval_to_enum(interval)
    response = await asyncio.to_thread(
        client.rest_api.klines,
        symbol=symbol,
        interval=interval_enum,
        **kwargs,
    )
    return _normalize_klines(response.data())


async def async_account(client: Spot):
    response = await asyncio.to_thread(client.rest_api.get_account, recv_window=10000)
    return response.data()


async def async_wallet_balances(client: Wallet):
    response = await asyncio.to_thread(client.rest_api.query_user_wallet_balance, recv_window=10000)
    return response.data()


async def async_new_order(client: Spot, **kwargs):
    payload = dict(kwargs)
    if "newClientOrderId" in payload:
        payload["new_client_order_id"] = payload.pop("newClientOrderId")
    if payload.get("new_client_order_id") is None:
        payload.pop("new_client_order_id", None)
    payload["side"] = NewOrderSideEnum(payload["side"])
    payload["type"] = NewOrderTypeEnum(payload["type"])
    payload.setdefault("recv_window", 10000)
    response = await asyncio.to_thread(client.rest_api.new_order, **payload)
    return response.data()


async def async_historical_klines(
    client: Spot,
    symbol: str,
    interval: str,
    start_str: str,
    end_str: str = "now",
    on_progress=None,
    on_chunk=None,
    on_retry=None,
    collect: bool = True,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    request_timeout: float | None = 15.0,
):
    start_ms = _parse_date(start_str)
    end_ms = _parse_date(end_str)
    step_ms = _interval_to_ms(interval)
    interval_enum = _interval_to_enum(interval)

    all_klines = []
    total = 0
    current = start_ms

    while current < end_ms:
        attempts = 0
        while True:
            try:
                request = asyncio.create_task(asyncio.to_thread(
                    client.rest_api.klines,
                    symbol=symbol,
                    interval=interval_enum,
                    start_time=current,
                    end_time=end_ms,
                    limit=KLINES_LIMIT,
                ))
                if request_timeout is None:
                    response = await request
                else:
                    try:
                        response = await asyncio.wait_for(asyncio.shield(request), timeout=request_timeout)
                    except asyncio.TimeoutError:
                        try:
                            await request
                        except Exception:
                            pass
                        raise
                break
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                attempts += 1
                if attempts > max_retries:
                    raise
                if on_retry:
                    result = on_retry(attempts, max_retries, current, exc)
                    if inspect.isawaitable(result):
                        await result
                if retry_delay > 0:
                    await asyncio.sleep(retry_delay * attempts)
        klines = response.data()
        if not klines:
            break
        normalized = _normalize_klines(klines)
        if collect:
            all_klines.extend(normalized)
        total += len(normalized)
        last_open_time = normalized[-1][0]
        current = last_open_time + step_ms
        if on_chunk:
            result = on_chunk(normalized, total, last_open_time, end_ms)
            if inspect.isawaitable(result):
                await result
        if on_progress:
            result = on_progress(total, last_open_time, end_ms)
            if inspect.isawaitable(result):
                await result

        if len(klines) < KLINES_LIMIT:
            break

    return all_klines


async def async_exchange_info(client: Spot, symbol: str):
    response = await asyncio.to_thread(client.rest_api.exchange_info, symbol=symbol)
    data = response.data()
    if hasattr(data, "to_dict"):
        return data.to_dict()
    return data


async def async_exchange_info_all(client: Spot):
    response = await asyncio.to_thread(client.rest_api.exchange_info)
    data = response.data()
    if hasattr(data, "to_dict"):
        return data.to_dict()
    return data


def _sync_time_offset(client: Spot):
    global _TIME_OFFSET_MS
    response = client.rest_api.time()
    data = response.data()
    server_time = getattr(data, "server_time", None)
    if server_time is None:
        raise ValueError("Unable to fetch server time from Binance.")
    local_time = int(time.time() * 1000)
    _TIME_OFFSET_MS = int(server_time) - local_time - 2000

    def _patched_timestamp():
        return int(time.time() * 1000) + _TIME_OFFSET_MS

    binance_utils.get_timestamp = _patched_timestamp


async def async_sync_time_offset(client: Spot) -> None:
    await asyncio.to_thread(_sync_time_offset, client)
