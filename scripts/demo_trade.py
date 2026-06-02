import asyncio
import pandas as pd
import torch
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from bat.config import conf
from bat.execution.broker import BinanceBroker
from bat.models.lstm import CryptoLSTM
from bat.data.dataset import DataProcessor
from bat.logger import install_crash_handler

async def run_demo():
    install_crash_handler()
    # 1. 初始化 Broker
    print("Demo trade is testnet-only; real trading is disabled for this script.")
    broker = BinanceBroker(is_testnet=True)
    await broker.init_client()

    # 2. 檢查帳戶 (測試網通常會送你一些 USDT)
    usdt_balance = await broker.get_balance('USDT')
    btc_balance = await broker.get_balance('BTC')
    print(f"當前餘額: {usdt_balance:.2f} USDT / {btc_balance:.6f} BTC")

    # 3. 獲取最新 K 線 (用來做決策)
    print("正在獲取即時市場數據...")
    klines = await broker.get_klines(
        symbol=conf.SYMBOL,
        interval=conf.INTERVAL,
        limit=conf.SEQ_LENGTH + 20 # 多抓一點做緩衝
    )

    # 轉 DataFrame
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'q_vol', 'trades', 'tb_base', 'tb_quote', 'ignore'
    ])

    # 4. 載入模型與預測 (假設你已經訓練好模型並存檔)
    model_path = "data/lstm_model.pth"
    if os.path.exists(model_path):
        processor = DataProcessor()
        # 預處理
        feature_data, _, _, _ = processor.process_for_training(df, conf.FEATURE_COLS)

        # 取最後 SEQ_LENGTH 筆作為輸入
        last_seq = feature_data[-conf.SEQ_LENGTH:]
        input_tensor = torch.FloatTensor(last_seq).unsqueeze(0).to(conf.DEVICE)

        # 載入模型架構
        model = CryptoLSTM(
            input_dim=len(conf.FEATURE_COLS),
            hidden_dim=conf.HIDDEN_SIZE,
            num_layers=conf.NUM_LAYERS
        ).to(conf.DEVICE)

        # 載入權重
        model.load_state_dict(torch.load(model_path, map_location=conf.DEVICE))
        model.eval()

        # 預測
        with torch.no_grad():
            probs = torch.softmax(model(input_tensor), dim=1).cpu().numpy()[0]

        current_price = float(df.iloc[-1]['close'])
        expected_return = (probs[2] - probs[0]) * conf.RETURN_THRESHOLD
        pred_price = current_price * (1.0 + expected_return)
        print(f"當前市價: {current_price}, AI預測: {pred_price:.2f}")

        # --- 策略邏輯 ---
        threshold = 1.001 # 預測漲幅 > 0.1% 才買

        if pred_price > current_price * threshold:
            print(">>> 訊號：看漲！準備買入...")
            if usdt_balance > 20: # 至少要有 20 U
                # 在測試網買入 20 USDT 等值的 BTC
                await broker.buy(quote_qty=20)
            else:
                print("餘額不足，無法買入")

        elif pred_price < current_price / threshold:
            print(">>> 訊號：看跌！準備賣出...")
            if btc_balance > 0.001:
                await broker.sell(quantity=0.001)
            else:
                print("持倉不足，無法賣出")
        else:
            print(">>> 訊號不明顯，HODL (持有觀望)")

    else:
        print("尚未找到訓練好的模型 (data/lstm_model.pth)，跳過預測階段。")

    await broker.close()

if __name__ == "__main__":
    asyncio.run(run_demo())
