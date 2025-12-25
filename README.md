# Binance Auto Trader (BAT)

BAT 是一個以 Python 實作的加密貨幣分析與自動交易系統，內建 RSI、Random Forest 與 LSTM 三種策略，並提供 TUI 介面用於監控與簡易操作。系統以 `asyncio` 搭配 `binance-sdk-spot` 與 `binance-sdk-wallet` 為核心，支援 Testnet 與實盤切換。

## ✨ 功能特色

- **多策略分析**：RSI、Random Forest、LSTM 同一入口切換。
- **TUI 操作介面**：側邊欄指令 + 日誌視窗即時輸出。
- **異步架構**：低延遲資料拉取與非阻塞 UI 互動。
- **測試網/實盤切換**：透過 `.env` 與 `TRADING_MODE` 控制。

## 🧭 專案結構

- `main.py`：主入口 (`--mode` 選擇策略或啟動 TUI)
- `src/bat/analyzer.py`：RSI / ML / LSTM 分析策略
- `src/bat/execution/broker.py`：交易執行與資產查詢
- `src/bat/tui/app.py`：Textual TUI 介面
- `scripts/train.py`：訓練 LSTM 模型
- `scripts/demo_trade.py`：以模型訊號做示範交易
- `scripts/download.py`：已停用（訓練資料改為 Testnet 模擬蒐集）
- `scripts/backtest.py`：回測 RSI 基準策略

## 🛠️ 安裝與環境設定

### 方法 A：使用 uv（建議）

```bash
uv sync
```

### 方法 B：使用 venv

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

> 注意：若不使用 `uv run` 或啟用 `.venv`，系統會找不到已安裝的依賴套件。

## 🔐 環境變數設定

在專案根目錄建立 `.env`：

```env
TRADING_MODE=TESTNET
TESTNET_API_KEY=your_testnet_key
TESTNET_API_SECRET=your_testnet_secret
BINANCE_API_KEY=your_real_key
BINANCE_API_SECRET=your_real_secret
```

- `TRADING_MODE` 預設為 `TESTNET`
- Testnet 與實盤 Key 請分開管理

## ▶️ 使用方式

### 啟動 TUI

```bash
python main.py
```

### 分析模式

```bash
python main.py --mode rsi
python main.py --mode ml
python main.py --mode lstm
```

### 訓練與示範交易（含回測與風控建議）

```bash
python scripts/train.py
python scripts/demo_trade.py
```

### 訓練資料蒐集說明

- 訓練資料只會透過 **Testnet 模擬交易** 產生
- K 線會寫入 `data/history.csv`
- 交易/決策會寫入 `data/testnet_trades.csv`

### 回測 RSI

```bash
python scripts/backtest.py
```

### 回測（含滑價/手續費/分段倉位/風控）

```bash
python scripts/backtest.py --fee 0.001 --slippage 0.0005 --splits 3 --stop-loss 0.03 --take-profit 0.06 --max-dd-stop 0.2
```

## 🩺 常見問題

- **ModuleNotFoundError: binance_sdk_spot / binance_sdk_wallet**：請確認已使用 `uv sync` 或啟用 `.venv`。
- **Testnet API 異常**：確認 `TRADING_MODE=TESTNET`，並且有設定 `TESTNET_API_KEY/SECRET`。
- **權限錯誤**：請確認 API Key 已開啟讀取/交易權限。

## 🧾 日誌

- 一般日誌：`logs/bat.log`
- 錯誤日誌：`logs/bat.error.log`
- 除錯日誌：`logs/bat.debug.log`
- 崩潰報告：`logs/crash_YYYYMMDD_HHMMSS.log`
- TUI 日誌會同步寫入檔案（與畫面顯示相同訊息）
- 日誌每日自動切檔並壓縮；也會在單日達到指定大小時分片壓縮（預設 100MB，可用 `BAT_LOG_MAX_MB` 調整）
