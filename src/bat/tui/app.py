import asyncio
import json
import os
import pandas as pd
import shutil
import subprocess
import torch
from datetime import datetime, timezone

from textual.app import App, ComposeResult
from textual.containers import Container
from textual.widgets import Button, Header, Footer, Static, RichLog, Label, Input, Select, Sparkline

from bat.config import conf
from bat.data.dataset import DataProcessor
from bat.execution.broker import BinanceBroker
from bat.execution.spot_client import async_exchange_info, async_exchange_info_all, async_historical_klines, create_spot_client
from bat.auto_trader import decide_trade, compute_order_size, apply_confidence_threshold
from bat.simulation import simulate_and_collect, append_kline, append_trade_event, write_klines
from bat.logger import get_logger, install_crash_handler
from bat.training import train_and_backtest, set_stop_training
from bat.training import should_stop_training
from bat.tui.line_chart import LineChart


class CryptoApp(App):
    CSS_PATH = "styles.tcss"
    BINDINGS = [("q", "quit", "Quit"), ("d", "toggle_dark", "Toggle Dark Mode")]

    auto_trading = False
    training_active = False
    price_series = []
    pred_series = []
    loss_series = []
    loss_ema_series = []
    balance_series = []
    equity_series = []
    train_status = {}
    settings_path = "data/settings.json"
    wallet_overview = []
    background_training_task = None
    last_train_ts = 0.0
    trade_events_since_train = 0
    warned_insufficient = False
    model_ready = False
    model_watch_task = None
    saved_symbol = None
    train_loop_active = False
    last_time_sync = 0.0
    current_tab = "trade"
    latest_balances = {}
    wallet_show_all = True
    wallet_sort_mode = "amount"
    winrate_ok = False
    winrate_series = []
    price_broker = None
    price_polling = False
    usdt_symbols = set()
    pretrain_done = False
    trained_rows = 0
    train_progress_path = "data/train_progress.json"

    def compose(self) -> ComposeResult:
        yield Header()

        with Container(id="sidebar"):
            yield Label("BAT 控制台", classes="title")
            yield Label("模式", classes="chart_title")
            yield Select(
                [("TESTNET", "TESTNET"), ("REAL", "REAL")],
                value=conf.TRADING_MODE,
                id="select_mode",
            )
            yield Label("交易對", classes="chart_title")
            yield Select([(conf.SYMBOL, conf.SYMBOL)], value=conf.SYMBOL, id="select_symbol")
            yield Label("K 線週期", classes="chart_title")
            yield Select(
                [("1m", "1m"), ("5m", "5m"), ("15m", "15m"), ("1h", "1h"), ("4h", "4h"), ("1d", "1d")],
                value=conf.INTERVAL,
                id="select_interval",
            )
            yield Label("交易輪詢間隔", classes="chart_title")
            yield Input(value="1m", placeholder="輪詢間隔，如 10s/1m/120", id="input_poll_interval")
            yield Label("模型信心門檻(%)", classes="chart_title")
            yield Input(value="55", placeholder="模型信心門檻(0-100)", id="input_confidence_threshold")
            yield Button("🚀 開始交易/查詢餘額", id="btn_paper", variant="success")
            yield Button("🤖 自動交易", id="btn_auto", variant="success")
            yield Button("🛑 停止/重置", id="btn_stop", variant="error")

            yield Static(self._mode_label_text(), id="mode_label", markup=True)
        with Container(id="content"):
            with Container(id="tab_bar"):
                yield Button("交易", id="btn_tab_trade", classes="tab_button", variant="primary")
                yield Button("錢包", id="btn_tab_wallet", classes="tab_button", variant="default")
                yield Button("訓練", id="btn_tab_train", classes="tab_button", variant="default")
                yield Button("圖表", id="btn_tab_charts", classes="tab_button", variant="default")
                yield Button("看盤", id="btn_tab_market", classes="tab_button", variant="default")

            with Container(id="tab_trade"):
                yield Label("交易資訊", classes="title")
                yield Static("", id="trade_info", markup=True)
                yield Label("成交/持倉明細", classes="chart_title")
                yield Static("", id="trade_records", markup=True)
                yield Label("系統日誌 (System Logs)", classes="title")
                yield RichLog(id="log_window", highlight=True, markup=True)

            with Container(id="tab_wallet"):
                yield Label("錢包資訊", classes="title")
                yield Button("顯示全部資產: 開", id="btn_wallet_filter", variant="primary")
                yield Button("排序: 餘額", id="btn_wallet_sort", variant="default")
                yield Label("賣出資產", classes="chart_title")
                yield Select([("USDT", "USDT")], value="USDT", id="select_sell_asset")
                yield Label("賣出數量", classes="chart_title")
                yield Input(value="0", placeholder="輸入數量或 all", id="input_sell_amount")
                yield Button("賣出", id="btn_sell_asset", variant="warning")
                yield Static("現價: -", id="sell_price", markup=True)
                yield Static("", id="wallet_info", markup=True)

            with Container(id="tab_train"):
                yield Label("訓練設定", classes="title")
                with Container(id="train_controls"):
                    with Container(id="train_controls_left"):
                        yield Label("模擬步數", classes="chart_title")
                        yield Input(value=str(conf.SIMULATION_STEPS), placeholder="模擬步數", id="input_sim_steps")
                        yield Label("顯存使用比例", classes="chart_title")
                        yield Input(value="0.5", placeholder="可用顯存比例 0.1-0.9", id="input_mem_fraction")
                        yield Label("Batch Size", classes="chart_title")
                        yield Input(value="64", placeholder="Batch Size (16-256)", id="input_batch_size")
                        yield Label("步進更新間隔", classes="chart_title")
                        yield Input(value="20", placeholder="步進更新間隔 (5-100)", id="input_status_every")
                    with Container(id="train_controls_right"):
                        yield Label("Online 訓練間隔秒", classes="chart_title")
                        yield Input(value="1800", placeholder="Online 訓練間隔秒", id="input_online_interval")
                        yield Label("Online 波動倍率", classes="chart_title")
                        yield Input(value="1.8", placeholder="Online 波動倍率", id="input_online_vol_mult")
                        yield Label("Online 波動冷卻秒", classes="chart_title")
                        yield Input(value="600", placeholder="Online 波動冷卻秒", id="input_online_vol_cooldown")
                        yield Label("Online 最少成交筆", classes="chart_title")
                        yield Input(value="1", placeholder="Online 最少成交筆", id="input_online_min_trades")
                yield Label("訓練動作", classes="chart_title")
                with Container(id="train_actions"):
                    yield Button("⬇️ 下載歷史資料", id="btn_download_history", variant="primary")
                    yield Button("🧪 開始模擬蒐集", id="btn_simulate", variant="primary")
                    yield Button("🧠 訓練模型", id="btn_train", variant="warning")
                    yield Button("🧹 重置訓練進度", id="btn_reset_train", variant="default")
                yield Label("訓練狀態", classes="chart_title")
                yield Static("", id="train_status")
                yield Label("訓練日誌", classes="title")
                yield RichLog(id="train_log", highlight=True, markup=True)

            with Container(id="tab_charts"):
                yield Label("圖表資訊", classes="title")
                with Container(id="charts"):
                    yield Label("價格走勢", classes="chart_title")
                    yield LineChart(id="chart_price")
                    yield Label("訓練 Loss", classes="chart_title")
                    yield LineChart(id="chart_loss")
                    yield Label("USDT 餘額", classes="chart_title")
                    yield Sparkline(id="chart_balance")
                    yield Label("回測淨值", classes="chart_title")
                    yield Sparkline(id="chart_equity")
                    yield Label("勝率(近100)", classes="chart_title")
                    yield LineChart(id="chart_winrate")

            with Container(id="tab_market"):
                yield Label("看盤資訊", classes="title")
                yield Static("", id="market_info", markup=True)

        yield Footer()

    def on_mount(self) -> None:
        install_crash_handler()
        self.logger = get_logger("bat.tui")
        self._load_settings()
        self.background_training_task = None
        self.last_train_ts = 0.0
        self.trade_events_since_train = 0
        self.warned_insufficient = False
        self.model_ready = False
        self.model_watch_task = None
        self.saved_symbol = None
        self.train_loop_active = False
        self.last_time_sync = 0.0
        self.pretrain_done = False
        self.trained_rows = self._load_trained_rows()
        self.log_msg("歡迎使用 Binance Auto Trader (BAT) v1.0")
        self.log_msg(f"目前交易對: [bold cyan]{conf.SYMBOL}[/]")
        self.log_msg(f"API 模式: {'[green]Testnet[/]' if conf.IS_TESTNET else '[bold red]REAL[/]'}")
        self._update_user_info({})
        self.run_worker(self.action_init_fetch(), exclusive=False)
        self.run_worker(self.action_load_symbols(), exclusive=False)
        self._show_tab("trade")
        self.set_interval(1.0, self._poll_sell_price)

    def log_msg(self, message: str) -> None:
        log_window = self.query_one("#log_window", RichLog)
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_window.write(f"[{timestamp}] {message}")
        if hasattr(self, "logger"):
            self.logger.info(message)

    def log_error(self, message: str, exc: Exception | None = None) -> None:
        log_window = self.query_one("#log_window", RichLog)
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_window.write(f"[{timestamp}] {message}")
        if hasattr(self, "logger"):
            if exc:
                self.logger.error(message, exc_info=exc)
            else:
                self.logger.error(message)

    def log_train(self, message: str) -> None:
        train_log = self.query_one("#train_log", RichLog)
        timestamp = datetime.now().strftime("%H:%M:%S")
        train_log.write(f"[{timestamp}] {message}")
        if hasattr(self, "logger"):
            self.logger.info(message)

    def log_train_error(self, message: str, exc: Exception | None = None) -> None:
        train_log = self.query_one("#train_log", RichLog)
        timestamp = datetime.now().strftime("%H:%M:%S")
        train_log.write(f"[{timestamp}] {message}")
        if hasattr(self, "logger"):
            if exc:
                self.logger.error(message, exc_info=exc)
            else:
                self.logger.error(message)

    def _log_train_threadsafe(self, message: str) -> None:
        try:
            self.call_from_thread(self.log_train, message)
        except Exception:
            self.log_train(message)

    def _log_train_error_threadsafe(self, message: str, exc: Exception | None = None) -> None:
        try:
            self.call_from_thread(self.log_train_error, message, exc)
        except Exception:
            self.log_train_error(message, exc)

    def _update_train_status_threadsafe(self, payload: dict) -> None:
        try:
            self.call_from_thread(self._update_train_status, payload)
        except Exception:
            self._update_train_status(payload)

    def _simulate_collect_sync(
        self,
        symbol: str,
        interval: str,
        steps: int,
        output_path: str,
        trade_log_path: str,
        poll_interval: str,
        confidence_threshold: float,
        progress_path: str,
        progress_key: str,
    ) -> int:
        async def runner() -> int:
            return await simulate_and_collect(
                symbol=symbol,
                interval=interval,
                steps=steps,
                output_path=output_path,
                trade_log_path=trade_log_path,
                is_testnet=True,
                on_status=self._update_train_status_threadsafe,
                on_log=self._log_train_threadsafe,
                poll_interval=poll_interval,
                confidence_threshold=confidence_threshold,
                progress_path=progress_path,
                progress_key=progress_key,
            )

        return asyncio.run(runner())

    async def on_button_pressed(self, event: Button.Pressed) -> None:
        btn_id = event.button.id

        if btn_id == "btn_tab_trade":
            self._show_tab("trade")
            return
        if btn_id == "btn_tab_wallet":
            self._show_tab("wallet")
            return
        if btn_id == "btn_tab_train":
            self._show_tab("train")
            return
        if btn_id == "btn_tab_charts":
            self._show_tab("charts")
            return
        if btn_id == "btn_tab_market":
            self._show_tab("market")
            return
        if btn_id == "btn_wallet_filter":
            self._toggle_wallet_filter()
            return
        if btn_id == "btn_wallet_sort":
            self._toggle_wallet_sort()
            return
        if btn_id == "btn_sell_asset":
            self.run_worker(self.action_sell_asset(), exclusive=True)
            return
        if btn_id == "btn_download_history":
            self.run_worker(self.action_download_history(), exclusive=True)
            return
        if btn_id == "btn_reset_train":
            self._reset_training_progress()
            return
        if btn_id == "btn_simulate":
            self.run_worker(self.action_simulate_data(), exclusive=True)
        elif btn_id == "btn_train":
            self.run_worker(self.action_train_model(), exclusive=True)
        elif btn_id == "btn_paper":
            self.run_worker(self.action_run_bot(), exclusive=True)
        elif btn_id == "btn_auto":
            self.run_worker(self.action_auto_trade(), exclusive=True)
        elif btn_id == "btn_stop":
            self.log_msg("[bold red]正在停止所有任務...[/]")
            self.auto_trading = False
            self.training_active = False
            self.train_loop_active = False
            set_stop_training(True)
            if self.background_training_task and not self.background_training_task.done():
                self.background_training_task.cancel()
            if self.model_watch_task and not self.model_watch_task.done():
                self.model_watch_task.cancel()

    async def action_simulate_data(self):
        symbol, interval, poll_interval, sim_steps, is_testnet = self._read_inputs()
        if not is_testnet:
            self.log_train_error("[bold red]❌ 模擬蒐集僅允許 Testnet[/]")
            return
        try:
            set_stop_training(False)
            count = await asyncio.to_thread(
                self._simulate_collect_sync,
                symbol,
                interval,
                sim_steps,
                "data/history.csv",
                "data/testnet_trades.csv",
                poll_interval,
                self._confidence_threshold(),
                "data/sim_progress.json",
                "simulate",
            )
            self.log_train(f"[bold green]✅ 模擬完成！共 {count} 筆[/]")
        except Exception as e:
            self.log_train_error(f"[bold red]❌ 模擬失敗: {e}[/]", e)
        finally:
            set_stop_training(False)
            await self._sync_time_offset()

    async def action_download_history(self) -> None:
        symbol, interval, _poll_interval, _sim_steps, is_testnet = self._read_inputs()
        try:
            if is_testnet:
                self.log_train("[歷史] Testnet 模式，改用主網公開資料下載")
            client = create_spot_client(is_testnet=False)
            info = await async_exchange_info(client, symbol)
            start_str = self._onboard_date_str(info)
            start_ms = self._parse_date_ms(start_str)
            end_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
            interval_ms = self._interval_ms_for_klines(interval)
            expected = max(int((end_ms - start_ms) / interval_ms), 1)
            self.log_train(f">>> [歷史] 下載 {symbol} {interval} 從 {start_str} 開始...")
            last_report = {"percent": 0.0, "rows": 0}

            def on_progress(total, current_ms=None, end_ms_value=None):
                if current_ms is not None and end_ms_value:
                    percent = min((current_ms - start_ms) / max(end_ms_value - start_ms, 1), 1.0)
                else:
                    percent = min(total / expected, 1.0)
                if percent - last_report["percent"] >= 0.01 or total - int(last_report.get("rows", 0)) >= 1000:
                    last_report["percent"] = percent
                    last_report["rows"] = total
                    bar = self._progress_bar(percent)
                    self.log_train(f">>> [歷史] {bar} 已下載 {total} 筆...")

            klines = await async_historical_klines(
                client,
                symbol,
                interval,
                start_str,
                "now",
                on_progress=on_progress,
            )
            self.log_train(">>> [歷史] 開始合併與寫入...")
            loop = asyncio.get_running_loop()
            last_merge = {"percent": 0.0}

            def on_merge_progress(done, total):
                if total <= 0:
                    return
                percent = min(done / total, 1.0)
                if percent - last_merge["percent"] < 0.01 and done % 100000 != 0:
                    return
                last_merge["percent"] = percent
                bar = self._progress_bar(percent)
                loop.call_soon_threadsafe(
                    self.log_train,
                    f">>> [歷史] 合併中 {bar} {done}/{total}",
                )

            count = await asyncio.to_thread(
                write_klines,
                "data/history.csv",
                klines,
                False,
                on_merge_progress,
            )
            self.log_train(f"[bold green]✅ 歷史資料下載完成（已合併）{count} 筆[/]")
        except Exception as exc:
            self.log_train_error(f"[bold red]❌ 歷史資料下載失敗: {exc}[/]", exc)

    async def action_train_model(self):
        if self.train_loop_active:
            self.log_train("[bold yellow]⚠️ 訓練循環已在執行中[/]")
            return
        symbol, interval, poll_interval, sim_steps, _is_testnet = self._read_inputs()
        self.log_train(">>> [訓練] 啟動收集→背景訓練循環...")
        try:
            set_stop_training(False)
            self.train_loop_active = True
            await self.action_download_history()
            total_count = self._history_count("data/history.csv")
            delta = max(total_count - self.trained_rows, 0)
            self.log_train(f">>> [訓練] 已檢測資料 {total_count} 筆 / 已訓練 {self.trained_rows} 筆 / 差異 {delta} 筆")
            if not self.pretrain_done:
                if total_count >= conf.SEQ_LENGTH and delta > 0:
                    self.log_train(f">>> [訓練] 先使用既有資料訓練 ({total_count} 筆)...")
                    await self._run_training_cycle()
                    self.pretrain_done = True
                    self.trained_rows = total_count
                    self._save_trained_rows(self.trained_rows)
                    if should_stop_training():
                        return
            while self.train_loop_active:
                symbol, interval, poll_interval, sim_steps, _ = self._read_inputs()
                await self._maybe_sync_time()
                if sim_steps < conf.SEQ_LENGTH:
                    sim_steps = conf.SEQ_LENGTH
                    self.query_one("#input_sim_steps", Input).value = str(sim_steps)
                    self.log_train(f"[bold yellow]⚠️ 模擬步數已提升為 {sim_steps}[/]")
                self._save_settings()
                self.log_train(f">>> [訓練] 新一輪收集開始 (目標 {sim_steps} 筆)...")
                before_count = self._history_count("data/history.csv")
                count = await asyncio.to_thread(
                    self._simulate_collect_sync,
                    symbol,
                    interval,
                    sim_steps,
                    "data/history.csv",
                    "data/testnet_trades.csv",
                    poll_interval,
                    self._confidence_threshold(),
                    "data/sim_progress.json",
                    "train_collect",
                )
                if should_stop_training():
                    break
                await self._maybe_sync_time()
                total_count = self._history_count("data/history.csv")
                new_rows = max(total_count - before_count, 0)
                self.log_train(f">>> [訓練] 本輪新增資料 {new_rows} 筆")
                delta = max(total_count - self.trained_rows, 0)
                self.log_train(f">>> [訓練] 累積差異 {delta} 筆")
                if total_count < conf.SEQ_LENGTH:
                    self.log_train_error(
                        f"[bold red]❌ 模擬資料不足({total_count}<{conf.SEQ_LENGTH})，繼續收集[/]"
                    )
                    continue
                if self.background_training_task and not self.background_training_task.done():
                    self.log_train("[bold yellow]⚠️ 背景訓練仍在執行，先繼續收集[/]")
                    continue
                if delta <= 0:
                    self.log_train("[bold yellow]⚠️ 無新增資料，跳過本輪訓練[/]")
                    continue
                self.log_train(f">>> [訓練] 啟動背景訓練 (累積 {total_count} 筆)...")
                self.background_training_task = asyncio.create_task(self._run_training_cycle())
        except Exception as e:
            self.log_train_error(f"[bold red]❌ 訓練失敗: {e}[/]", e)
        finally:
            self.train_loop_active = False
            set_stop_training(False)
            await self._sync_time_offset()

    async def _run_training_cycle(self):
        try:
            self.training_active = True
            mem_fraction = self._get_mem_fraction_value()
            os.environ["BAT_GPU_MEM_FRACTION"] = f"{mem_fraction:.2f}"
            os.environ["BAT_BATCH_SIZE"] = str(self._get_batch_size_value())
            monitor_task = asyncio.create_task(self._monitor_gpu_usage())
            loop = asyncio.get_running_loop()

            def on_epoch_loss(avg_loss):
                loop.call_soon_threadsafe(self._append_loss, avg_loss)

            def on_status(payload):
                loop.call_soon_threadsafe(self._update_train_status, payload)

            result, risk = await asyncio.to_thread(
                train_and_backtest,
                on_epoch_loss=on_epoch_loss,
                on_status=on_status,
                status_every=self._get_status_every_value(),
            )
            if result.equity_curve:
                self._set_equity_series(result.equity_curve)
            self.log_train("[bold green]✅ 背景訓練完成[/]")
            self.log_train(
                f">>> 回測總報酬: {result.total_return:.2%} | "
                f"最大回撤: {result.max_drawdown:.2%} | "
                f"虧損率: {result.loss_rate:.2%}"
            )
            self.log_train(
                f">>> 風控建議: SL {risk.stop_loss:.2%} / "
                f"TP {risk.take_profit:.2%} / "
                f"MaxDD {risk.max_dd_stop:.2%} / "
                f"Splits {risk.position_splits}"
            )
            total_count = self._history_count("data/history.csv")
            self.trained_rows = total_count
            self._save_trained_rows(self.trained_rows)
        except Exception as e:
            self.log_train_error(f"[bold red]❌ 背景訓練失敗: {e}[/]", e)
        finally:
            self.training_active = False
            set_stop_training(False)

    async def action_run_bot(self):
        self.log_msg(">>> [交易] 正在連接 Binance API...")
        btn = self.query_one("#btn_paper", Button)
        btn.disabled = True
        try:
            await self._sync_time_offset()
            _, _, _, _, is_testnet = self._read_inputs()
            broker = BinanceBroker(is_testnet=is_testnet)
            await broker.init_client()
            balances = await self._fetch_balances(broker)
            self.log_msg("[bold green]✅ 連線成功！[/]")
            for asset, amount in balances.items():
                display = f"{amount:.6f}" if amount < 1 else f"{amount:.2f}"
                self.log_msg(f"💰 [bold yellow]{asset} 餘額: {display}[/]")
            if self.wallet_overview:
                for item in self.wallet_overview:
                    name = item.get("walletName", "Wallet")
                    balance = item.get("balance", "0")
                    self.log_msg(f"🏦 [bold cyan]{name} 餘額: {balance}[/]")
            self._update_user_info(balances)
        except Exception as e:
            self.log_error(f"[bold red]❌ 錯誤: {e}[/]", e)
        finally:
            if "broker" in locals() and broker.client:
                await broker.close()
            btn.disabled = False

    async def action_auto_trade(self):
        symbol, interval, poll_interval, _, is_testnet = self._read_inputs()
        base_asset, quote_asset = self._split_symbol(symbol)
        self.auto_trading = True
        self.log_msg(f">>> [自動] 啟動自動交易 ({symbol}, {interval})")

        broker = BinanceBroker(is_testnet=is_testnet)
        await broker.init_client()
        try:
            await self._sync_time_offset()
            if self.model_watch_task is None or self.model_watch_task.done():
                self.model_watch_task = asyncio.create_task(self._watch_model_ready())
            last_logged_close_time = None
            while self.auto_trading:
                balances = await self._fetch_balances(broker, [base_asset, quote_asset])
                self._update_user_info(balances)

                klines = await broker.get_klines(symbol=symbol, interval=interval, limit=conf.SEQ_LENGTH + 50)
                if len(klines) < conf.SEQ_LENGTH + 1:
                    if not self.warned_insufficient:
                        self.log_error("[自動] 市場資料不足，等待更多 K 線...")
                        self.warned_insufficient = True
                    await asyncio.sleep(self._interval_seconds(poll_interval))
                    continue
                self.warned_insufficient = False
                decision, risk = decide_trade(klines)
                if decision is None or risk is None:
                    self.log_error("[自動] 無法取得交易決策，跳過本輪")
                else:
                    decision = apply_confidence_threshold(decision, self._confidence_threshold())
                    if decision.source == "filtered":
                        self.log_msg(
                            f"[自動] 信心不足 {decision.confidence:.2%} < "
                            f"{self._confidence_threshold():.0%}，改為觀望"
                        )
                    if is_testnet and klines:
                        latest = klines[-1]
                        close_time = int(latest[6])
                        if last_logged_close_time != close_time:
                            append_kline("data/history.csv", latest)
                            last_logged_close_time = close_time
                    self._append_price(decision.current_price, decision.predicted_price)
                    self._append_balance(balances.get(quote_asset, 0.0))
                    self._update_market_info(klines)
                    self._update_trade_records()
                    market_vol = self._calc_market_vol(klines)
                    invest = compute_order_size(balances.get(quote_asset, 0.0), market_vol)
                    decision.invest_amount = invest
                    action_label = {"BUY": "買入", "SELL": "賣出", "HOLD": "觀望"}.get(decision.action, decision.action)
                    source_label = {"model": "模型", "fallback": "簡易"}.get(decision.source, decision.source)
                    self.log_msg(
                        f"[自動] {action_label}({source_label}) | 訊號 {decision.signal} | 現價 {decision.current_price:.4f} | "
                        f"預測 {decision.predicted_price:.4f} | 信心 {decision.confidence:.2%}"
                    )
                    if risk is not None:
                        self.log_msg(
                            f"[自動] 參數 投入={decision.invest_amount:.4f} "
                            f"止損={risk.stop_loss:.2%} 止盈={risk.take_profit:.2%} "
                            f"最大回撤={risk.max_dd_stop:.2%} 分段={risk.position_splits}"
                        )
                    order = None
                    if decision.action == "BUY" and invest > 0:
                        order = await broker.buy(quote_qty=invest)
                    elif decision.action == "SELL" and balances.get(base_asset, 0.0) > 0:
                        sell_qty = balances[base_asset] * min(max(market_vol * 5.0, 0.1), 1.0)
                        order = await broker.sell(quantity=sell_qty)
                    if is_testnet:
                        append_trade_event(
                            "data/testnet_trades.csv",
                            [
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                symbol,
                                action_label,
                                source_label,
                                f"{decision.confidence:.6f}",
                                f"{decision.current_price:.6f}",
                                f"{decision.predicted_price:.6f}",
                                f"{(decision.predicted_price - decision.current_price) / decision.current_price:.6%}"
                                if decision.current_price
                                else "0.000000%",
                                f"{self._actual_return_from_klines(klines):.6%}",
                                f"{decision.invest_amount:.6f}",
                                f"{market_vol:.6f}",
                                f"{risk.stop_loss:.6f}" if risk else "",
                                f"{risk.take_profit:.6f}" if risk else "",
                                f"{risk.max_dd_stop:.6f}" if risk else "",
                                f"{risk.position_splits}" if risk else "",
                                f"{balances.get(base_asset, 0.0):.6f}",
                                f"{balances.get(quote_asset, 0.0):.6f}",
                                self._order_field(order, "order_id"),
                                self._order_field(order, "status"),
                                self._order_field(order, "executed_qty"),
                                self._order_field(order, "cummulative_quote_qty"),
                            ],
                        )
                        if order is not None:
                            self.trade_events_since_train += 1

                await self._maybe_trigger_online_training(interval, klines, is_testnet)
                await asyncio.sleep(self._interval_seconds(poll_interval))
        finally:
            if self.model_watch_task and not self.model_watch_task.done():
                self.model_watch_task.cancel()
            await broker.close()
            self.log_msg("[自動] 已停止自動交易")

    async def action_init_fetch(self):
        symbol, interval, _, _, is_testnet = self._read_inputs()
        self.log_msg("[初始化] 取得餘額與市場資料...")
        broker = BinanceBroker(is_testnet=is_testnet)
        await broker.init_client()
        try:
            balances = await self._fetch_balances(broker)
            self._update_user_info(balances)
            klines = await broker.get_klines(symbol=symbol, interval=interval, limit=2)
            if klines:
                last_price = float(klines[-1][4])
                self.log_msg(f"市場資料：{symbol} 最新價 {last_price:.4f}")
                self._append_price(last_price)
                self._update_market_info(klines)
            self._update_trade_records()
            self._append_balance(balances.get("USDT", 0.0))
        except Exception as e:
            self.log_error(f"[bold red]❌ 初始化失敗: {e}[/]", e)
        finally:
            await broker.close()

    async def action_load_symbols(self):
        _, _, _, _, is_testnet = self._read_inputs()
        client = create_spot_client(is_testnet=is_testnet)
        try:
            self.log_msg(">>> 正在更新交易對清單...")
            info = await async_exchange_info_all(client)
            symbols = []
            for item in info.get("symbols", []) if isinstance(info, dict) else []:
                symbol = item.get("symbol")
                status = item.get("status")
                if symbol and status == "TRADING" and symbol.endswith("USDT"):
                    symbols.append(symbol)
            symbols = sorted(set(symbols))
            self.usdt_symbols = set(symbols)
            if symbols:
                options = [(s, s) for s in symbols]
                select = self.query_one("#select_symbol", Select)
                select.set_options(options)
                target = self.saved_symbol or self._read_saved_symbol_value() or select.value
                if target in symbols:
                    select.value = target
                    self.saved_symbol = target
                else:
                    select.value = symbols[0]
            self._refresh_sell_asset_options()
            self.log_msg(f">>> 交易對更新完成，共 {len(symbols)} 筆")
        except Exception as exc:
            self.log_error(f"[bold red]❌ 取得交易對失敗: {exc}[/]", exc)
        finally:
            try:
                await self._sync_time_offset()
            except Exception:
                pass

    def _mode_label_text(self):
        try:
            is_testnet = self.query_one("#select_mode", Select).value == "TESTNET"
        except Exception:
            is_testnet = conf.IS_TESTNET
        mode_color = "green" if is_testnet else "red"
        mode_text = "TESTNET" if is_testnet else "REAL MONEY"
        return f"\n模式: [{mode_color}]{mode_text}[/]"

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "select_mode":
            label = self.query_one("#mode_label", Static)
            label.update(self._mode_label_text())
            self.run_worker(self.action_load_symbols(), exclusive=False)
            self.run_worker(self.action_init_fetch(), exclusive=False)
            self.run_worker(self._close_price_broker(), exclusive=False)
        elif event.select.id == "select_symbol":
            self.saved_symbol = event.value
        self._save_settings()

    def on_input_changed(self, event: Input.Changed) -> None:
        if not event.value.strip():
            return
        # Avoid normalizing while typing to prevent cursor jumping.
        return

    def on_input_blurred(self, event: Input.Blurred) -> None:
        self._normalize_input(event.input.id, event.value)
        self._save_settings()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self._normalize_input(event.input.id, event.value)
        self._save_settings()

    def _read_inputs(self):
        symbol = self.query_one("#select_symbol", Select).value or conf.SYMBOL
        interval = self.query_one("#select_interval", Select).value or conf.INTERVAL
        poll_interval = self.query_one("#input_poll_interval", Input).value.strip() or "1m"
        sim_steps = self._get_sim_steps_value()
        mem_value = self.query_one("#input_mem_fraction", Input).value.strip()
        mode_value = self.query_one("#select_mode", Select).value or "TESTNET"
        is_testnet = mode_value == "TESTNET"
        if mem_value:
            self._set_mem_fraction_input(mem_value)
        status_value = self.query_one("#input_status_every", Input).value.strip()
        if status_value:
            self._set_status_every_input(status_value)
        batch_value = self.query_one("#input_batch_size", Input).value.strip()
        if batch_value:
            self._set_batch_size_input(batch_value)
        return symbol or conf.SYMBOL, interval, poll_interval, sim_steps, is_testnet

    def _normalize_input(self, input_id: str | None, value: str) -> None:
        if not input_id:
            return
        if input_id == "input_mem_fraction":
            self._set_mem_fraction_input(value)
        elif input_id == "input_status_every":
            self._set_status_every_input(value)
        elif input_id == "input_batch_size":
            self._set_batch_size_input(value)
        elif input_id == "input_poll_interval":
            self.query_one("#input_poll_interval", Input).value = value.strip() or "1m"
        elif input_id == "input_sim_steps":
            self.query_one("#input_sim_steps", Input).value = str(self._get_sim_steps_value())
        elif input_id == "input_online_interval":
            self._set_online_interval_input(value)
        elif input_id == "input_online_vol_mult":
            self._set_online_vol_mult_input(value)
        elif input_id == "input_online_vol_cooldown":
            self._set_online_vol_cooldown_input(value)
        elif input_id == "input_online_min_trades":
            self._set_online_min_trades_input(value)
        elif input_id == "input_confidence_threshold":
            self._set_confidence_threshold_input(value)

    async def _fetch_balances(self, broker: BinanceBroker, assets=None):
        assets = assets or conf.DEFAULT_ASSETS
        balances = {}
        for asset in assets:
            try:
                amount = await broker.get_balance(asset)
                balances[asset] = amount
            except Exception as exc:
                if "recvWindow" in str(exc) or "Timestamp" in str(exc):
                    await self._sync_time_offset()
                    try:
                        amount = await broker.get_balance(asset)
                        balances[asset] = amount
                        continue
                    except Exception:
                        pass
                self.log_error(f"❌ 無法取得 {asset} 餘額", exc)
        try:
            wallet_data = await broker.get_wallet_overview()
            self.wallet_overview = [
                item.to_dict() if hasattr(item, "to_dict") else item for item in wallet_data
            ]
        except Exception:
            self.wallet_overview = []
        return balances

    def _update_user_info(self, balances: dict):
        try:
            mode_value = self.query_one("#select_mode", Select).value
        except Exception:
            mode_value = conf.TRADING_MODE
        self.latest_balances = balances or {}
        symbol = self.query_one("#select_symbol", Select).value or conf.SYMBOL
        base_asset, quote_asset = self._split_symbol(symbol)
        trade_lines = [
            f"使用者: [bold]{conf.USER_NAME}[/]",
            f"模式: {mode_value}",
            f"交易對: {symbol}",
            f"{base_asset}: {self._format_balance(self.latest_balances.get(base_asset, 0.0))}",
            f"{quote_asset}: {self._format_balance(self.latest_balances.get(quote_asset, 0.0))}",
        ]
        winrate_line = self._winrate_summary_line()
        if winrate_line:
            trade_lines.append(winrate_line)
        wallet_lines = [
            f"使用者: [bold]{conf.USER_NAME}[/]",
            f"模式: {mode_value}",
        ]
        if self.latest_balances:
            items = [
                (asset, amount)
                for asset, amount in self.latest_balances.items()
                if self.wallet_show_all or float(amount or 0.0) != 0.0
            ]
            if self.wallet_sort_mode == "alpha":
                items.sort(key=lambda item: item[0])
            else:
                items.sort(key=lambda item: float(item[1] or 0.0), reverse=True)
            for asset, amount in items:
                wallet_lines.append(f"{asset}: {self._format_balance(amount)}")
        if self.wallet_overview:
            wallet_lines.append("錢包:")
            for item in self.wallet_overview:
                name = item.get("walletName", "Wallet")
                balance = item.get("balance", "0")
                wallet_lines.append(f"- {name}: {balance}")
        elif not self.latest_balances:
            wallet_lines.append("餘額: 尚未連線")
        self.query_one("#trade_info", Static).update("\n".join(trade_lines))
        self.query_one("#wallet_info", Static).update("\n".join(wallet_lines))
        self._refresh_sell_asset_options()

    def _user_info_text(self, balances: dict) -> str:
        return f"使用者: [bold]{conf.USER_NAME}[/]\n模式: {conf.TRADING_MODE}\n餘額: 尚未連線"

    def _append_price(self, value, predicted=None):
        self.price_series.append(float(value))
        self.price_series = self.price_series[-120:]
        if predicted is None:
            self.pred_series.append(None)
        else:
            self.pred_series.append(float(predicted))
        self.pred_series = self.pred_series[-120:]
        self._update_price_chart()

    def _append_loss(self, value):
        self.loss_series.append(float(value))
        self.loss_series = self.loss_series[-120:]
        if not self.loss_ema_series:
            self.loss_ema_series.append(float(value))
        else:
            alpha = 0.3
            self.loss_ema_series.append(alpha * float(value) + (1 - alpha) * self.loss_ema_series[-1])
        self.loss_ema_series = self.loss_ema_series[-120:]
        self._update_loss_chart()

    def _append_balance(self, value):
        self.balance_series.append(float(value))
        self.balance_series = self.balance_series[-120:]
        self._update_chart("chart_balance", self.balance_series)

    def _set_equity_series(self, values):
        self.equity_series = [float(v) for v in values][-120:]
        self._update_chart("chart_equity", self.equity_series)

    def _update_chart(self, chart_id, values):
        try:
            chart = self.query_one(f"#{chart_id}", Sparkline)
            chart.data = values
        except Exception:
            pass

    def _update_price_chart(self) -> None:
        try:
            chart = self.query_one("#chart_price", LineChart)
            chart.set_series(
                {
                    "price": (self.price_series, "cyan", "●"),
                    "pred": (self.pred_series, "magenta", "●"),
                }
            )
        except Exception:
            pass

    def _update_loss_chart(self) -> None:
        try:
            chart = self.query_one("#chart_loss", LineChart)
            chart.set_series(
                {
                    "loss": (self.loss_series, "yellow", "●"),
                    "ema": (self.loss_ema_series, "green", "●"),
                }
            )
        except Exception:
            pass

    def _update_market_info(self, klines) -> None:
        if not klines:
            return
        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "q_vol", "trades", "tb_base", "tb_quote", "ignore"
        ])
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        last_price = float(df.iloc[-1]["close"])
        prev_price = float(df.iloc[-2]["close"]) if len(df) > 1 else last_price
        change_pct = (last_price - prev_price) / prev_price if prev_price else 0.0
        high = float(df["high"].max())
        low = float(df["low"].min())
        vol_sum = float(df["volume"].sum())
        processor = DataProcessor()
        df_ind = processor.add_indicators(df)
        rsi = float(df_ind.iloc[-1]["RSI"]) if not df_ind.empty else 0.0
        ema = float(df_ind.iloc[-1]["EMA_20"]) if not df_ind.empty else 0.0
        ret1 = float(df_ind.iloc[-1]["RET_1"]) if not df_ind.empty else 0.0
        vol20 = float(df_ind.iloc[-1]["VOL_20"]) if not df_ind.empty else 0.0
        volz = float(df_ind.iloc[-1]["VOL_Z"]) if not df_ind.empty else 0.0
        macd_line = signal_line = hist = 0.0
        bb_upper = bb_middle = bb_lower = 0.0
        vwap = 0.0
        try:
            macd = df.ta.macd(close="close")
            if macd is not None and not macd.empty:
                macd_line = float(macd.iloc[-1][0])
                signal_line = float(macd.iloc[-1][1])
                hist = float(macd.iloc[-1][2])
        except Exception:
            pass
        try:
            bbands = df.ta.bbands(close="close")
            if bbands is not None and not bbands.empty:
                bb_lower = float(bbands.iloc[-1][0])
                bb_middle = float(bbands.iloc[-1][1])
                bb_upper = float(bbands.iloc[-1][2])
        except Exception:
            pass
        try:
            vwap_series = df.ta.vwap(high="high", low="low", close="close", volume="volume")
            if vwap_series is not None and not vwap_series.empty:
                vwap = float(vwap_series.iloc[-1])
        except Exception:
            pass
        symbol = self.query_one("#select_symbol", Select).value or conf.SYMBOL
        interval = self.query_one("#select_interval", Select).value or conf.INTERVAL
        lines = [
            f"交易對: [bold]{symbol}[/]  週期: {interval}",
            f"現價: {last_price:.8f}  漲跌: {change_pct:.2%}",
            f"區間高低: {high:.8f} / {low:.8f}",
            f"成交量合計: {vol_sum:.8f}",
            f"RSI: {rsi:.2f}  EMA20: {ema:.8f}",
            f"RET_1: {ret1:.4%}  VOL_20: {vol20:.6f}  VOL_Z: {volz:.2f}",
            f"MACD: {macd_line:.6f}  Signal: {signal_line:.6f}  Hist: {hist:.6f}",
            f"BBands: 上 {bb_upper:.8f} 中 {bb_middle:.8f} 下 {bb_lower:.8f}",
            f"VWAP: {vwap:.8f}",
        ]
        self.query_one("#market_info", Static).update("\n".join(lines))

    def _show_tab(self, tab_name: str) -> None:
        self.current_tab = tab_name
        trade = self.query_one("#tab_trade", Container)
        wallet = self.query_one("#tab_wallet", Container)
        train = self.query_one("#tab_train", Container)
        charts = self.query_one("#tab_charts", Container)
        market = self.query_one("#tab_market", Container)
        trade.display = tab_name == "trade"
        wallet.display = tab_name == "wallet"
        train.display = tab_name == "train"
        charts.display = tab_name == "charts"
        market.display = tab_name == "market"
        btn_trade = self.query_one("#btn_tab_trade", Button)
        btn_wallet = self.query_one("#btn_tab_wallet", Button)
        btn_train = self.query_one("#btn_tab_train", Button)
        btn_charts = self.query_one("#btn_tab_charts", Button)
        btn_market = self.query_one("#btn_tab_market", Button)
        btn_trade.variant = "primary" if tab_name == "trade" else "default"
        btn_wallet.variant = "primary" if tab_name == "wallet" else "default"
        btn_train.variant = "primary" if tab_name == "train" else "default"
        btn_charts.variant = "primary" if tab_name == "charts" else "default"
        btn_market.variant = "primary" if tab_name == "market" else "default"

    def _format_balance(self, amount: float) -> str:
        try:
            return f"{float(amount):.8f}"
        except Exception:
            return "0.00000000"

    def _toggle_wallet_filter(self) -> None:
        self.wallet_show_all = not self.wallet_show_all
        self._update_wallet_filter_button()
        self._update_user_info(self.latest_balances)
        self._save_settings()

    def _update_wallet_filter_button(self) -> None:
        label = "顯示全部資產: 開" if self.wallet_show_all else "顯示全部資產: 關"
        variant = "primary" if self.wallet_show_all else "default"
        button = self.query_one("#btn_wallet_filter", Button)
        button.label = label
        button.variant = variant

    def _toggle_wallet_sort(self) -> None:
        self.wallet_sort_mode = "alpha" if self.wallet_sort_mode == "amount" else "amount"
        self._update_wallet_sort_button()
        self._update_user_info(self.latest_balances)
        self._save_settings()

    def _update_wallet_sort_button(self) -> None:
        label = "排序: 餘額" if self.wallet_sort_mode == "amount" else "排序: 字母"
        button = self.query_one("#btn_wallet_sort", Button)
        button.label = label

    def _refresh_sell_asset_options(self) -> None:
        assets = sorted(self.latest_balances.keys()) if self.latest_balances else ["USDT"]
        valid_assets = []
        for asset in assets:
            if asset == "USDT":
                valid_assets.append(asset)
            elif not self.usdt_symbols or f"{asset}USDT" in self.usdt_symbols:
                valid_assets.append(asset)
        if not valid_assets:
            valid_assets = ["USDT"]
        options = [(a, a) for a in valid_assets]
        select = self.query_one("#select_sell_asset", Select)
        current = select.value
        select.set_options(options)
        if current in valid_assets and current != Select.BLANK:
            select.value = current
        else:
            select.value = valid_assets[0]

    def _poll_sell_price(self) -> None:
        if self.price_polling:
            return
        self.price_polling = True
        self.run_worker(self._update_sell_price(), exclusive=False)

    async def _update_sell_price(self) -> None:
        try:
            asset = self.query_one("#select_sell_asset", Select).value
            if not asset or asset == Select.BLANK:
                self.query_one("#sell_price", Static).update("現價: -")
                return
            if asset == "USDT":
                self.query_one("#sell_price", Static).update("現價: 1.00000000")
                return
            if self.usdt_symbols and f"{asset}USDT" not in self.usdt_symbols:
                self.query_one("#sell_price", Static).update("現價: -")
                return
            _, _, _, _, is_testnet = self._read_inputs()
            if not self.price_broker or self.price_broker.is_testnet != is_testnet:
                if self.price_broker:
                    await self.price_broker.close()
                self.price_broker = BinanceBroker(is_testnet=is_testnet)
                await self.price_broker.init_client()
            symbol = f"{asset}USDT"
            klines = await self.price_broker.get_klines(symbol=symbol, interval="1m", limit=1)
            if klines:
                price = float(klines[-1][4])
                self.query_one("#sell_price", Static).update(f"現價: {price:.8f}")
        except Exception as exc:
            self.log_error(f"[賣出] 取得現價失敗: {exc}", exc)
        finally:
            self.price_polling = False

    async def _close_price_broker(self) -> None:
        if self.price_broker:
            await self.price_broker.close()
            self.price_broker = None
    def _update_trade_records(self) -> None:
        path = "data/testnet_trades.csv"
        if not os.path.exists(path):
            self.query_one("#trade_records", Static).update("尚無成交紀錄")
            return
        try:
            lines = []
            with open(path, "r", encoding="utf-8") as handle:
                rows = handle.read().strip().splitlines()
            if len(rows) <= 1:
                self.query_one("#trade_records", Static).update("尚無成交紀錄")
                return
            header = rows[0].split(",")
            key_map = {
                "時間戳": "timestamp",
                "交易對": "symbol",
                "動作": "action",
                "現價": "price",
                "預測價": "pred_price",
                "預期收益": "expected",
                "實際收益": "actual",
                "投入金額": "invest",
                "訂單ID": "order_id",
                "狀態": "status",
                "成交數量": "executed_qty",
                "成交金額": "quote_qty",
            }
            for row in rows[-30:]:
                cols = row.split(",")
                data = dict(zip(header, cols))
                normalized = {key_map.get(k, k): v for k, v in data.items()}
                ts = normalized.get("timestamp", "")
                action = normalized.get("action", "")
                if action in ("觀望", "HOLD"):
                    continue
                order_id = normalized.get("order_id", "")
                status = normalized.get("status", "")
                price = normalized.get("price", "")
                qty = normalized.get("invest", "")
                symbol = normalized.get("symbol", "")
                expected = normalized.get("expected", "")
                actual = normalized.get("actual", "")
                executed = normalized.get("executed_qty", "")
                quote = normalized.get("quote_qty", "")
                if not order_id and not status:
                    continue
                lines.append(
                    f"{ts} {symbol} {action} 價格={price} 投入={qty} "
                    f"預期={expected} 實際={actual} 成交={executed} 金額={quote}"
                )
            self.query_one("#trade_records", Static).update("\n".join(lines) or "尚無成交紀錄")
            self._update_winrate_status(rows)
        except Exception:
            self.query_one("#trade_records", Static).update("成交紀錄讀取失敗")

    def _reset_training_progress(self) -> None:
        checkpoint_path = "data/lstm_checkpoint.pth"
        removed = False
        if os.path.exists(checkpoint_path):
            try:
                os.remove(checkpoint_path)
                removed = True
            except Exception:
                pass
        if removed:
            self.log_msg("[訓練] 已清除 checkpoint，下一次訓練將重新開始")
        else:
            self.log_msg("[訓練] 無可清除的 checkpoint")
        self.pretrain_done = False

    async def action_sell_asset(self) -> None:
        asset = self.query_one("#select_sell_asset", Select).value
        amount_str = self.query_one("#input_sell_amount", Input).value.strip()
        if not asset or asset == "USDT":
            self.log_error("[賣出] 請選擇非 USDT 資產")
            return
        balance = float(self.latest_balances.get(asset, 0.0))
        if amount_str.lower() == "all":
            amount = balance
        else:
            try:
                amount = float(amount_str)
            except ValueError:
                self.log_error("[賣出] 數量格式錯誤")
                return
        if amount <= 0:
            self.log_error("[賣出] 數量必須大於 0")
            return
        if amount > balance:
            self.log_error("[賣出] 數量超過可用餘額")
            return
        _, _, _, _, is_testnet = self._read_inputs()
        broker = BinanceBroker(is_testnet=is_testnet)
        await broker.init_client()
        try:
            broker.symbol = f"{asset}USDT"
            order = await broker.sell(quantity=amount)
            if order is not None:
                self.log_msg(f"[賣出] 已送出 {asset} 賣單: {amount}")
            else:
                self.log_error("[賣出] 送單失敗")
        finally:
            await broker.close()

    def _winrate_summary_line(self) -> str:
        stats = self._compute_win_rate("data/testnet_trades.csv", window=100)
        overall = self._compute_win_rate("data/testnet_trades.csv", window=None)
        if not stats and not overall:
            return ""
        lines = []
        if stats:
            rate = stats["rate"]
            if 0.4 <= rate <= 0.6:
                band = "[green]區間內[/]"
            else:
                band = "[yellow]區間外[/]"
            lines.append(
                f"近100勝率: {rate:.2%} ({stats['wins']}/{stats['total']}) {band}"
            )
        if overall:
            lines.append(f"總勝率: {overall['rate']:.2%} ({overall['wins']}/{overall['total']})")
        return " | ".join(lines)

    def _update_winrate_status(self, rows: list[str]) -> None:
        stats = self._compute_win_rate_rows(rows, window=100)
        if not stats:
            return
        target_ok = stats["rate"] >= 0.75 and stats["total"] >= 50
        if target_ok and not self.winrate_ok:
            self.log_msg("[風控] ✅ 近100勝率達標 (>=75%)")
        self.winrate_ok = target_ok
        self._append_winrate(stats["rate"])

    def _compute_win_rate(self, path: str, window: int | None = None):
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                rows = handle.read().strip().splitlines()
            return self._compute_win_rate_rows(rows, window=window)
        except Exception:
            return None

    def _compute_win_rate_rows(self, rows: list[str], window: int | None = None):
        if len(rows) <= 1:
            return None
        header = rows[0].split(",")
        key_map = {
            "動作": "action",
            "實際收益": "actual",
        }
        data_rows = rows[1:]
        if window:
            data_rows = data_rows[-window:]
        wins = 0
        total = 0
        for row in data_rows:
            cols = row.split(",")
            data = dict(zip(header, cols))
            normalized = {key_map.get(k, k): v for k, v in data.items()}
            action = normalized.get("action", "")
            actual = normalized.get("actual", "")
            if action in ("觀望", "HOLD") or not actual:
                continue
            try:
                actual_val = float(actual)
            except ValueError:
                continue
            if action in ("買入", "BUY"):
                win = actual_val > 0
            elif action in ("賣出", "SELL"):
                win = actual_val < 0
            else:
                continue
            total += 1
            if win:
                wins += 1
        if total == 0:
            return None
        return {"wins": wins, "total": total, "rate": wins / total}

    def _append_winrate(self, rate: float) -> None:
        try:
            value = float(rate)
        except Exception:
            return
        self.winrate_series.append(value)
        self.winrate_series = self.winrate_series[-120:]
        self._update_winrate_chart()

    def _update_winrate_chart(self) -> None:
        try:
            chart = self.query_one("#chart_winrate", LineChart)
            lower = [0.4 for _ in self.winrate_series]
            upper = [0.6 for _ in self.winrate_series]
            chart.set_series(
                {
                    "winrate": (self.winrate_series, "cyan", "●"),
                    "lower": (lower, "green", "·"),
                    "upper": (upper, "green", "·"),
                }
            )
        except Exception:
            pass

    def _history_count(self, path: str) -> int:
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return max(sum(1 for _ in handle) - 1, 0)
        except Exception:
            return 0

    def _load_trained_rows(self) -> int:
        try:
            with open(self.train_progress_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
            return int(data.get("trained_rows", 0))
        except Exception:
            return 0

    def _save_trained_rows(self, rows: int) -> None:
        os.makedirs(os.path.dirname(self.train_progress_path) or ".", exist_ok=True)
        payload = {"trained_rows": int(rows), "updated_at": datetime.now().isoformat()}
        with open(self.train_progress_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)

    def _parse_date_ms(self, value: str) -> int:
        if not value:
            return int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        if value.lower() == "now":
            return int(datetime.now(tz=timezone.utc).timestamp() * 1000)
        dt = pd.to_datetime(value, utc=True)
        return int(dt.timestamp() * 1000)

    def _interval_ms_for_klines(self, interval: str) -> int:
        interval = (interval or "").strip()
        if not interval:
            return 60 * 1000
        if interval[-1].isdigit():
            return max(int(interval) * 60 * 1000, 1000)
        unit = interval[-1]
        value = int(interval[:-1])
        if unit == "s":
            return max(value * 1000, 1000)
        if unit == "m":
            return max(value * 60 * 1000, 1000)
        if unit == "h":
            return max(value * 60 * 60 * 1000, 1000)
        if unit == "d":
            return max(value * 24 * 60 * 60 * 1000, 1000)
        if unit == "w":
            return max(value * 7 * 24 * 60 * 60 * 1000, 1000)
        if unit == "M":
            return max(value * 30 * 24 * 60 * 60 * 1000, 1000)
        return 60 * 1000

    def _onboard_date_str(self, info: dict) -> str:
        onboard_ms = None
        if isinstance(info, dict):
            symbol_info = None
            if info.get("symbols"):
                symbol_info = info["symbols"][0]
            else:
                symbol_info = info
            if isinstance(symbol_info, dict):
                onboard_ms = symbol_info.get("onboardDate")
        if onboard_ms:
            try:
                dt = datetime.utcfromtimestamp(int(onboard_ms) / 1000)
                return dt.strftime("%Y-%m-%d %H:%M:%S")
            except Exception:
                pass
        return "2017-07-01 00:00:00"

    def _get_sim_steps_value(self) -> int:
        value = self.query_one("#input_sim_steps", Input).value.strip()
        try:
            num = int(value)
        except ValueError:
            num = conf.SIMULATION_STEPS
        return max(10, min(num, 10000))

    def _get_mem_fraction_value(self) -> float:
        value = self.query_one("#input_mem_fraction", Input).value.strip()
        try:
            num = float(value)
        except ValueError:
            num = 0.5
        return min(max(num, 0.1), 0.9)

    def _set_mem_fraction_input(self, value: str):
        try:
            mem_fraction = float(value)
        except ValueError:
            mem_fraction = 0.5
        mem_fraction = min(max(mem_fraction, 0.1), 0.9)
        self.query_one("#input_mem_fraction", Input).value = f"{mem_fraction:.2f}"

    def _load_settings(self):
        try:
            with open(self.settings_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            data = {}
        mode_value = data.get("mode", conf.TRADING_MODE)
        self.query_one("#select_mode", Select).value = mode_value
        symbol_value = data.get("symbol", conf.SYMBOL)
        self.saved_symbol = symbol_value
        select_symbol = self.query_one("#select_symbol", Select)
        try:
            select_symbol.value = symbol_value
        except Exception:
            select_symbol.value = conf.SYMBOL
        interval_value = data.get("interval", conf.INTERVAL)
        self.query_one("#select_interval", Select).value = interval_value
        poll_value = data.get("poll_interval", "1m")
        self.query_one("#input_poll_interval", Input).value = str(poll_value)
        sim_steps = int(data.get("sim_steps", conf.SIMULATION_STEPS))
        self.query_one("#input_sim_steps", Input).value = str(sim_steps)
        mem_fraction = data.get("gpu_mem_fraction", 0.5)
        self.query_one("#input_mem_fraction", Input).value = f"{float(mem_fraction):.2f}"
        online_interval = int(data.get("online_train_interval", 1800))
        self._set_online_interval_input(str(online_interval))
        online_vol_mult = float(data.get("online_train_vol_mult", 1.8))
        self._set_online_vol_mult_input(str(online_vol_mult))
        online_vol_cooldown = int(data.get("online_train_vol_cooldown", 600))
        self._set_online_vol_cooldown_input(str(online_vol_cooldown))
        online_min_trades = int(data.get("online_train_min_trades", 1))
        self._set_online_min_trades_input(str(online_min_trades))
        status_every = int(data.get("status_every", 20))
        self._set_status_every_input(str(status_every))
        batch_size = int(data.get("batch_size", conf.BATCH_SIZE))
        self._set_batch_size_input(str(batch_size))
        confidence = float(data.get("confidence_threshold", 0.55))
        self._set_confidence_threshold_input(str(confidence * 100))
        self.wallet_show_all = bool(data.get("wallet_show_all", True))
        self._update_wallet_filter_button()
        self.wallet_sort_mode = data.get("wallet_sort_mode", "amount")
        if self.wallet_sort_mode not in ("amount", "alpha"):
            self.wallet_sort_mode = "amount"
        self._update_wallet_sort_button()

    def _read_saved_symbol_value(self):
        try:
            with open(self.settings_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            return None
        return data.get("symbol")

    def _save_settings(self):
        data = {
            "mode": self.query_one("#select_mode", Select).value,
            "symbol": self.query_one("#select_symbol", Select).value or conf.SYMBOL,
            "interval": self.query_one("#select_interval", Select).value or conf.INTERVAL,
            "poll_interval": self.query_one("#input_poll_interval", Input).value.strip() or "1m",
            "sim_steps": self._get_sim_steps_value(),
            "gpu_mem_fraction": self._get_mem_fraction_value(),
            "online_train_interval": self._get_online_interval_value(),
            "online_train_vol_mult": self._get_online_vol_mult_value(),
            "online_train_vol_cooldown": self._get_online_vol_cooldown_value(),
            "online_train_min_trades": self._get_online_min_trades_value(),
            "status_every": self._get_status_every_value(),
            "batch_size": self._get_batch_size_value(),
            "confidence_threshold": self._confidence_threshold(),
            "wallet_show_all": self.wallet_show_all,
            "wallet_sort_mode": self.wallet_sort_mode,
        }
        os.makedirs(os.path.dirname(self.settings_path), exist_ok=True)
        with open(self.settings_path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=True, indent=2)

    def _get_status_every_value(self) -> int:
        value = self.query_one("#input_status_every", Input).value.strip()
        try:
            num = int(value)
        except ValueError:
            num = 20
        return max(5, min(num, 100))

    def _get_batch_size_value(self) -> int:
        value = self.query_one("#input_batch_size", Input).value.strip()
        try:
            num = int(value)
        except ValueError:
            num = conf.BATCH_SIZE
        return max(16, min(num, 256))

    def _set_status_every_input(self, value: str) -> None:
        try:
            num = int(value)
        except ValueError:
            num = 20
        num = max(5, min(num, 100))
        self.query_one("#input_status_every", Input).value = str(num)

    def _set_batch_size_input(self, value: str) -> None:
        try:
            num = int(value)
        except ValueError:
            num = conf.BATCH_SIZE
        num = max(16, min(num, 256))
        self.query_one("#input_batch_size", Input).value = str(num)

    def _confidence_threshold(self) -> float:
        value = self.query_one("#input_confidence_threshold", Input).value.strip()
        try:
            num = float(value)
        except ValueError:
            num = 55.0
        return max(0.0, min(num, 100.0)) / 100.0

    def _set_confidence_threshold_input(self, value: str) -> None:
        try:
            num = float(value)
        except ValueError:
            num = 55.0
        num = max(0.0, min(num, 100.0))
        self.query_one("#input_confidence_threshold", Input).value = f"{num:.0f}"

    def _passes_confidence(self, decision) -> bool:
        return decision.confidence >= self._confidence_threshold()

    def _get_online_interval_value(self) -> int:
        value = self.query_one("#input_online_interval", Input).value.strip()
        try:
            num = int(value)
        except ValueError:
            num = 1800
        return max(60, min(num, 86400))

    def _get_online_vol_mult_value(self) -> float:
        value = self.query_one("#input_online_vol_mult", Input).value.strip()
        try:
            num = float(value)
        except ValueError:
            num = 1.8
        return max(1.0, min(num, 5.0))

    def _get_online_vol_cooldown_value(self) -> int:
        value = self.query_one("#input_online_vol_cooldown", Input).value.strip()
        try:
            num = int(value)
        except ValueError:
            num = 600
        return max(60, min(num, 3600))

    def _get_online_min_trades_value(self) -> int:
        value = self.query_one("#input_online_min_trades", Input).value.strip()
        try:
            num = int(value)
        except ValueError:
            num = 1
        return max(0, min(num, 1000))

    def _set_online_interval_input(self, value: str) -> None:
        try:
            num = int(value)
        except ValueError:
            num = 1800
        num = max(60, min(num, 86400))
        self.query_one("#input_online_interval", Input).value = str(num)

    def _set_online_vol_mult_input(self, value: str) -> None:
        try:
            num = float(value)
        except ValueError:
            num = 1.8
        num = max(1.0, min(num, 5.0))
        self.query_one("#input_online_vol_mult", Input).value = f"{num:.2f}"

    def _set_online_vol_cooldown_input(self, value: str) -> None:
        try:
            num = int(value)
        except ValueError:
            num = 600
        num = max(60, min(num, 3600))
        self.query_one("#input_online_vol_cooldown", Input).value = str(num)

    def _set_online_min_trades_input(self, value: str) -> None:
        try:
            num = int(value)
        except ValueError:
            num = 1
        num = max(0, min(num, 1000))
        self.query_one("#input_online_min_trades", Input).value = str(num)

    def _interval_seconds(self, interval: str) -> int:
        interval = (interval or "").strip()
        if not interval:
            return 60
        if interval[-1].isdigit():
            return max(1, min(int(interval), 300))
        unit = interval[-1]
        value = int(interval[:-1])
        if unit == "s":
            return max(1, min(value, 300))
        if unit == "m":
            return max(1, min(value * 60, 300))
        if unit == "h":
            return max(1, min(value * 3600, 300))
        if unit == "d":
            return max(1, min(value * 86400, 300))
        return 300

    def _calc_market_vol(self, klines) -> float:
        df = pd.DataFrame(klines, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "q_vol", "trades", "tb_base", "tb_quote", "ignore"
        ])
        df["close"] = df["close"].astype(float)
        returns = df["close"].pct_change().dropna()
        return float(returns.std()) if not returns.empty else 0.0

    def _actual_return_from_klines(self, klines) -> float:
        if not klines or len(klines) < 2:
            return 0.0
        prev_price = float(klines[-2][4])
        current_price = float(klines[-1][4])
        if prev_price == 0:
            return 0.0
        return (current_price - prev_price) / prev_price

    def _interval_minutes(self, interval: str) -> int:
        unit = interval[-1]
        value = int(interval[:-1])
        if unit == "m":
            return value
        if unit == "h":
            return value * 60
        if unit == "d":
            return value * 1440
        return 1

    async def _maybe_trigger_online_training(self, interval: str, klines, is_testnet: bool) -> None:
        if not is_testnet:
            return
        if self.training_active:
            return
        if self.background_training_task and not self.background_training_task.done():
            return
        now = datetime.now().timestamp()
        elapsed = now - self.last_train_ts if self.last_train_ts else None
        if self.trade_events_since_train < self._online_train_min_trades():
            return
        interval_sec = self._online_train_interval_seconds()
        if elapsed is None or elapsed >= interval_sec:
            await self._start_online_training("time")
            return
        if klines:
            interval_min = self._interval_minutes(interval)
            recent_window = max(5, int(15 / max(interval_min, 1)))
            baseline_window = max(10, int(360 / max(interval_min, 1)))
            if len(klines) >= baseline_window:
                recent_vol = self._calc_volatility_window(klines, recent_window)
                baseline_vol = self._calc_volatility_window(klines, baseline_window)
                if baseline_vol > 0:
                    threshold = baseline_vol * self._online_train_vol_multiplier()
                    if recent_vol > threshold and elapsed >= self._online_train_vol_cooldown():
                        await self._start_online_training("volatility")

    def _calc_volatility_window(self, klines, window: int) -> float:
        if not klines or window <= 1:
            return 0.0
        recent = klines[-window:]
        df = pd.DataFrame(recent, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "q_vol", "trades", "tb_base", "tb_quote", "ignore"
        ])
        df["close"] = df["close"].astype(float)
        returns = df["close"].pct_change().dropna()
        return float(returns.std()) if not returns.empty else 0.0

    def _online_train_interval_seconds(self) -> int:
        return self._get_online_interval_value()

    def _online_train_vol_multiplier(self) -> float:
        return self._get_online_vol_mult_value()

    def _online_train_vol_cooldown(self) -> int:
        return self._get_online_vol_cooldown_value()

    def _online_train_min_trades(self) -> int:
        return self._get_online_min_trades_value()

    async def _start_online_training(self, reason: str) -> None:
        self.last_train_ts = datetime.now().timestamp()
        self.trade_events_since_train = 0
        self.log_train(f"[Online Train] 觸發訓練 ({reason})")
        self.background_training_task = asyncio.create_task(self._run_online_training())

    async def _run_online_training(self) -> None:
        set_stop_training(False)
        self.training_active = True
        mem_fraction = self._get_mem_fraction_value()
        os.environ["BAT_GPU_MEM_FRACTION"] = f"{mem_fraction:.2f}"
        os.environ["BAT_BATCH_SIZE"] = str(self._get_batch_size_value())
        monitor_task = asyncio.create_task(self._monitor_gpu_usage())
        loop = asyncio.get_running_loop()

        def on_epoch_loss(avg_loss):
            loop.call_soon_threadsafe(self._append_loss, avg_loss)

        def on_status(payload):
            loop.call_soon_threadsafe(self._update_train_status, payload)

        try:
            result, _risk = await asyncio.to_thread(
                train_and_backtest,
                on_epoch_loss=on_epoch_loss,
                on_status=on_status,
                status_every=self._get_status_every_value(),
            )
            if result.equity_curve:
                self._set_equity_series(result.equity_curve)
            self.log_train("[Online Train] 完成")
        except Exception as exc:
            self.log_train_error(f"[Online Train] 失敗: {exc}")
        finally:
            self.training_active = False
            await monitor_task
            set_stop_training(False)

    async def _monitor_gpu_usage(self):
        if not torch.cuda.is_available():
            return
        while self.training_active:
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            self.log_train(f"GPU 記憶體: allocated={allocated:.1f}MB reserved={reserved:.1f}MB")
            if shutil.which("nvidia-smi"):
                output = await asyncio.to_thread(
                    subprocess.check_output,
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.total",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                )
                self._update_train_status({"gpu_util": output.strip()})
                self.log_train(f"GPU 使用率: {output.strip()}")
            await asyncio.sleep(5)

    async def _sync_time_offset(self):
        try:
            client = create_spot_client(is_testnet=conf.IS_TESTNET)
            await asyncio.to_thread(client.rest_api.time)
        except Exception as exc:
            self.log_error("[bold red]❌ 時間同步失敗[/]", exc)

    async def _maybe_sync_time(self):
        now = datetime.now().timestamp()
        if now - self.last_time_sync >= 300:
            await self._sync_time_offset()
            self.last_time_sync = now

    async def _watch_model_ready(self) -> None:
        model_path = "data/lstm_model.pth"
        last_state = None
        while self.auto_trading:
            exists = os.path.exists(model_path)
            state = "ready" if exists else "missing"
            if state != last_state:
                self._update_train_status({"model_status": state})
                if state == "ready":
                    self.log_msg("[模型] 已偵測到模型，開始使用模型決策")
                else:
                    self.log_msg("[模型] 尚未偵測到模型，使用簡易規則下單")
                last_state = state
            await asyncio.sleep(1)

    def _update_train_status(self, payload: dict):
        self.train_status.update(payload)
        parts = []
        if self.train_status.get("model_status"):
            parts.append(f"Model: {self.train_status['model_status']}")
        if self.train_status.get("mode"):
            parts.append(f"Mode: {self.train_status['mode']}")
        if self.train_status.get("samples") is not None:
            parts.append(f"Samples: {self.train_status['samples']}")
        if self.train_status.get("collect_count") is not None and self.train_status.get("collect_total") is not None:
            parts.append(f"Collect: {self.train_status['collect_count']}/{self.train_status['collect_total']}")
        if self.train_status.get("gpu_util"):
            parts.append(f"GPU: {self.train_status['gpu_util']}")
        if self.train_status.get("progress") is not None:
            parts.append(f"Total {self._progress_bar(self.train_status['progress'])}")
        if self.train_status.get("chunk_progress") is not None:
            parts.append(f"Chunk {self._progress_bar(self.train_status['chunk_progress'])}")
        status_text = " | ".join(parts) if parts else "準備中"
        try:
            self.query_one("#train_status", Static).update(status_text)
        except Exception:
            pass

    def _progress_bar(self, value: float) -> str:
        value = min(max(value, 0.0), 1.0)
        width = 20
        filled = int(width * value)
        bar = "#" * filled + "-" * (width - filled)
        return f"[{bar}] {value * 100:.0f}%"

    def _order_field(self, order, name):
        if order is None:
            return None
        if isinstance(order, dict):
            return order.get(name)
        return getattr(order, name, None)

    def _split_symbol(self, symbol: str):
        if symbol.endswith("USDT"):
            return symbol[:-4], "USDT"
        return symbol[:-3], symbol[-3:]


if __name__ == "__main__":
    app = CryptoApp()
    app.run()
