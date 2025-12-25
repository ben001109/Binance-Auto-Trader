# 📝 Binance Auto Trader (BAT) - Development Log

本文件記錄專案重要里程碑與主要模組演進，避免與實際程式碼脫節。

---

## 📌 專案概述

- **名稱**: Binance Auto Trader (BAT)
- **目標**: 建立模組化、可擴充的加密貨幣分析與交易系統
- **核心技術**: Python, binance-sdk-spot, binance-sdk-wallet, pandas-ta, scikit-learn, PyTorch, Textual

---

## 📅 里程碑

### Phase 1: 基礎分析能力
- 建立 RSI 策略與 K 線資料擷取
- `src/bat/analyzer.py` 作為統一策略入口

### Phase 2: 機器學習策略
- 引入 Random Forest 進行漲跌方向判斷
- 完成特徵工程與簡易訓練流程

### Phase 3: 深度學習策略
- 加入 LSTM 預測價格
- 訓練流程與 `data/lstm_model.pth` 權重輸出

### Phase 4: 交易與執行封裝
- `src/bat/execution/broker.py` 封裝 Binance API
- 支援 Testnet / 實盤切換

### Phase 5: TUI 介面
- Textual 介面提供操作與日誌視窗
- 非同步 Worker 避免 UI 阻塞

---

## 🧭 目前核心模組

- `main.py`：啟動 TUI 或 CLI 分析模式
- `src/bat/analyzer.py`：RSI / ML / LSTM 策略
- `src/bat/execution/broker.py`：交易執行與餘額查詢
- `src/bat/tui/app.py`：TUI 介面
- `scripts/train.py`：LSTM 訓練
- `scripts/demo_trade.py`：示範交易流程
- `scripts/download.py`：歷史資料下載

---

## ✅ 近期調整記錄

- Testnet 連線統一使用 `testnet=conf.IS_TESTNET`
- 增加 API Key/Secret 缺失時的前置檢查
- 修正文檔內容與現有程式碼不一致的問題
