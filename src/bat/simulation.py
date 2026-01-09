import csv
import json
import os
import statistics
import time
from typing import Iterable

from bat.config import conf
from bat.execution.broker import BinanceBroker
from bat.logger import get_logger
from bat.analyzer import AnalystAgent, compute_order_size, apply_confidence_threshold
from bat.training import should_stop_training


KLINE_HEADERS = [
    "timestamp",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "q_vol",
    "trades",
    "tb_base",
    "tb_quote",
    "ignore",
]
TRADE_HEADERS = [
    "時間戳",
    "交易對",
    "動作",
    "模式",
    "信心度",
    "現價",
    "預測價",
    "預期收益",
    "實際收益",
    "投入金額",
    "市場波動",
    "止損",
    "止盈",
    "最大回撤",
    "分段倉位",
    "基礎資產餘額",
    "報價資產餘額",
    "訂單ID",
    "狀態",
    "成交數量",
    "成交金額",
]


def _ensure_csv(path: str, headers: Iterable[str]) -> None:
    if os.path.exists(path) and os.path.getsize(path) > 0:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                first_line = handle.readline().strip()
            existing = [item.strip() for item in first_line.split(",") if item.strip()]
            if existing == list(headers):
                return
            backup = f"{path}.bak.{int(time.time())}"
            os.rename(path, backup)
        except Exception:
            return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(headers))


def _append_row(path: str, row: Iterable) -> None:
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(row))


def _order_field(order, name, fallback=None):
    if isinstance(order, dict):
        return order.get(name, fallback)
    return getattr(order, name, fallback)


def _calc_market_vol(klines) -> float:
    if not klines or len(klines) < 3:
        return 0.0
    closes = []
    for row in klines:
        try:
            closes.append(float(row[4]))
        except (ValueError, TypeError, IndexError):
            continue
    if len(closes) < 3:
        return 0.0
    returns = []
    for i in range(1, len(closes)):
        if closes[i - 1] == 0:
            continue
        returns.append((closes[i] - closes[i - 1]) / closes[i - 1])
    if len(returns) < 2:
        return 0.0
    return float(statistics.stdev(returns))


def _action_label(action: str) -> str:
    return {
        "BUY": "買入",
        "SELL": "賣出",
        "HOLD": "觀望",
    }.get(action, action)


def _source_label(source: str) -> str:
    return {
        "model": "模型",
        "fallback": "簡易",
        "filtered": "信心不足",
    }.get(source, source)


async def simulate_and_collect(
    symbol: str,
    interval: str,
    steps: int,
    output_path: str,
    trade_log_path: str,
    is_testnet: bool = True,
    on_status=None,
    on_log=None,
    log_every: int = 10,
    poll_interval: str | None = None,
    confidence_threshold: float | None = None,
    progress_path: str | None = None,
    progress_key: str = "simulate",
) -> int:
    logger = get_logger("bat.simulation")
    if not is_testnet:
        raise ValueError("Training data collection must run on Testnet.")

    _ensure_csv(output_path, KLINE_HEADERS)
    _ensure_csv(trade_log_path, TRADE_HEADERS)

    logger.info(
        "Simulation start: symbol=%s interval=%s steps=%s",
        symbol,
        interval,
        steps,
    )
    if on_log:
        on_log(f">>> [模擬] 開始蒐集 {symbol} {interval}，目標 {steps} 筆")

    broker = BinanceBroker(is_testnet=True)
    await broker.init_client()
    
    # Initialize Analyst Agent
    agent = AnalystAgent(client=broker.client, mode='lstm', symbol=symbol, interval=interval)

    last_close_time = None
    collected = 0
    if progress_path:
        collected = _load_progress(progress_path, progress_key, symbol, interval, steps)
        if collected > 0 and on_log:
            on_log(f">>> [模擬] 續接進度: {collected}/{steps}")
    if on_status:
        on_status({
            "collect_count": collected,
            "collect_total": steps,
            "progress": collected / max(steps, 1)
        })

    try:
        while collected < steps:
            if should_stop_training():
                logger.warning("Simulation stopped by user")
                break

            klines = await broker.get_klines(
                symbol=symbol,
                interval=interval,
                limit=conf.SEQ_LENGTH + 50,
            )
            if not klines:
                await _sleep_interval(interval, fallback=2)
                continue

            latest = klines[-1]
            close_time = int(latest[6])
            if last_close_time == close_time:
                await _sleep_interval(interval, fallback=2)
                continue

            last_close_time = close_time
            append_kline(output_path, latest)
            collected += 1
            if on_status:
                on_status({
                    "collect_count": collected,
                    "collect_total": steps,
                    "progress": collected / max(steps, 1)
                })
            if progress_path:
                _save_progress(progress_path, progress_key, symbol, interval, steps, collected)
            if collected % max(log_every, 1) == 0:
                logger.info("Simulation progress: %s/%s", collected, steps)
                if on_log:
                    on_log(f">>> [模擬] 蒐集中: {collected}/{steps}")

                if on_log:
                    on_log(f">>> [模擬] 蒐集中: {collected}/{steps}")

            decision, risk = await agent.analyze(klines)
            if decision is None:
                await _sleep_interval(interval, fallback=2)
                continue
            if confidence_threshold is not None:
                decision = apply_confidence_threshold(decision, confidence_threshold)

            base_asset, quote_asset = _split_symbol(symbol)
            base_balance = await broker.get_balance(base_asset)
            quote_balance = await broker.get_balance(quote_asset)
            market_vol = _calc_market_vol(klines)
            decision.invest_amount = compute_order_size(quote_balance, market_vol)
            expected_return = (
                (decision.predicted_price - decision.current_price) / decision.current_price
                if decision.current_price
                else 0.0
            )
            actual_return = 0.0
            if len(klines) > 1:
                prev_price = float(klines[-2][4])
                if prev_price:
                    actual_return = (decision.current_price - prev_price) / prev_price

            order = None
            if decision.action == "BUY" and decision.invest_amount > 0:
                order = await broker.buy(quote_qty=decision.invest_amount)
            elif decision.action == "SELL" and base_balance > 0:
                order = await broker.sell(quantity=base_balance)

            append_trade_event(
                trade_log_path,
                [
                    _format_timestamp_ms(int(time.time() * 1000)),
                    symbol,
                    _action_label(decision.action),
                    _source_label(decision.source),
                    f"{decision.confidence:.6f}",
                    f"{decision.current_price:.6f}",
                    f"{decision.predicted_price:.6f}",
                    f"{expected_return:.6%}",
                    f"{actual_return:.6%}",
                    f"{decision.invest_amount:.6f}",
                    f"{market_vol:.6f}",
                    f"{risk.stop_loss:.6f}" if risk else "",
                    f"{risk.take_profit:.6f}" if risk else "",
                    f"{risk.max_dd_stop:.6f}" if risk else "",
                    f"{risk.position_splits}" if risk else "",
                    f"{base_balance:.6f}",
                    f"{quote_balance:.6f}",
                    _order_field(order, "order_id"),
                    _order_field(order, "status"),
                    _order_field(order, "executed_qty"),
                    _order_field(order, "cummulative_quote_qty"),
                ],
            )
            if on_log:
                on_log(
                    ">>> [模擬交易] "
                    f"{_action_label(decision.action)}({_source_label(decision.source)}) "
                    f"訊號={decision.signal} "
                    f"投入={decision.invest_amount:.4f} "
                    f"現價={decision.current_price:.4f} 預測={decision.predicted_price:.4f} "
                    f"信心={decision.confidence:.2%} 預期={expected_return:.2%} 實際={actual_return:.2%} "
                    f"止損={risk.stop_loss:.2%} 止盈={risk.take_profit:.2%} "
                    f"最大回撤={risk.max_dd_stop:.2%} 分段={risk.position_splits}"
                )

            await _sleep_interval(poll_interval or interval)
    finally:
        await broker.close()

    logger.info("Simulation complete: collected=%s", collected)
    if on_log:
        on_log(f">>> [模擬] 完成，實際蒐集 {collected} 筆")
    if progress_path and collected >= steps:
        _clear_progress(progress_path, progress_key)
    return collected


async def _sleep_interval(interval: str, fallback: int = 5) -> None:
    unit = interval[-1]
    try:
        value = int(interval[:-1])
    except ValueError:
        value = 0
    if unit == "m":
        await _sleep_seconds(value * 60)
    elif unit == "h":
        await _sleep_seconds(value * 3600)
    elif unit == "d":
        await _sleep_seconds(value * 86400)
    else:
        await _sleep_seconds(fallback)


async def _sleep_seconds(seconds: int) -> None:
    import asyncio

    await asyncio.sleep(max(seconds, 1))


def _split_symbol(symbol: str):
    if symbol.endswith("USDT"):
        return symbol[:-4], "USDT"
    return symbol[:-3], symbol[-3:]


def _format_timestamp_ms(ts_ms: int) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts_ms / 1000))
    except Exception:
        return str(ts_ms)


def _load_progress(path: str, key: str, symbol: str, interval: str, steps: int) -> int:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        data = {}
    record = data.get(key, {})
    if (
        record.get("symbol") == symbol
        and record.get("interval") == interval
        and int(record.get("steps", 0)) == int(steps)
    ):
        return int(record.get("collected", 0))
    return 0


def _save_progress(path: str, key: str, symbol: str, interval: str, steps: int, collected: int) -> None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        data = {}
    data[key] = {
        "symbol": symbol,
        "interval": interval,
        "steps": int(steps),
        "collected": int(collected),
        "updated_at": _format_timestamp_ms(int(time.time() * 1000)),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=True, indent=2)


def _clear_progress(path: str, key: str) -> None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return
    if key in data:
        data.pop(key, None)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=True, indent=2)
def append_kline(path: str, kline: Iterable) -> None:
    _ensure_csv(path, KLINE_HEADERS)
    _append_row(path, kline)


def append_trade_event(path: str, row: Iterable) -> None:
    _ensure_csv(path, TRADE_HEADERS)
    _append_row(path, row)


def write_klines(path: str, klines: Iterable[Iterable], overwrite: bool = True, on_progress=None) -> int:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    count = 0
    if overwrite:
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(list(KLINE_HEADERS))
            for row in klines:
                writer.writerow(list(row))
                count += 1
    else:
        existing = {}
        if os.path.exists(path) and os.path.getsize(path) > 0:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    reader = csv.reader(handle)
                    headers = next(reader, [])
                    if headers != list(KLINE_HEADERS):
                        _ensure_csv(path, KLINE_HEADERS)
                    else:
                        for row in reader:
                            if not row:
                                continue
                            existing[row[0]] = row
            except Exception:
                existing = {}
        for row in klines:
            if not row:
                continue
            existing[str(row[0])] = list(row)
        merged = sorted(existing.values(), key=lambda item: int(item[0]))
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(list(KLINE_HEADERS))
            for idx, row in enumerate(merged, start=1):
                writer.writerow(row)
                if on_progress and idx % 100000 == 0:
                    on_progress(idx, len(merged))
        count = len(merged)
    return count
