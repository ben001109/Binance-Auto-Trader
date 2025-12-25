import asyncio
import numpy as np
import pandas as pd
import pandas_ta as ta
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

from bat.execution.spot_client import async_klines, create_spot_client
from bat.logger import get_logger
# 假設你有 config.py，若無可直接填入 .env 讀取邏輯
try:
    from bat.config import conf
except ImportError:
    # Fallback 如果沒有 config 模組
    from dotenv import load_dotenv
    load_dotenv()
    class Config:
        API_KEY = os.getenv('BINANCE_API_KEY')
        API_SECRET = os.getenv('BINANCE_API_SECRET')
        SYMBOL = 'BTCUSDT'
        INTERVAL = '15m'
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    conf = Config()

# ==========================================
# 1. 深度學習模型定義 (LSTM)
# ==========================================
class CryptoLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1):
        super(CryptoLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(conf.DEVICE)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(conf.DEVICE)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# ==========================================
# 2. 策略基類 (Base Strategy)
# ==========================================
class BaseStrategy:
    # 預設值還是可以用 conf，但現在允許覆蓋 (override)
    def __init__(self, client, symbol=conf.SYMBOL, interval=conf.INTERVAL):
        self.client = client
        self.symbol = symbol      # 使用傳入的參數
        self.interval = interval  # 使用傳入的參數

    async def fetch_data(self, limit=100):
        print(f"[{self.__class__.__name__}] 正在獲取 {self.symbol} ({self.interval}) 數據...")
        logger = get_logger("bat.analyzer")
        logger.info("Fetch data: %s %s limit=%s", self.symbol, self.interval, limit)

        klines = await async_klines(
            self.client,
            self.symbol,
            self.interval,
            limit=limit
        )

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'q_vol', 'trades', 'tb_base', 'tb_quote', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        cols = ['open', 'high', 'low', 'close', 'volume']
        df[cols] = df[cols].astype(float)
        return df

    async def analyze(self):
        raise NotImplementedError("子類別必須實作 analyze 方法")

# ==========================================
# 3. 傳統技術指標策略 (RSI)
# ==========================================
class RSIStrategy(BaseStrategy):
    async def analyze(self):
        df = await self.fetch_data(limit=100)

        # 計算 RSI
        df['rsi'] = df.ta.rsi(length=14)
        last_row = df.iloc[-1]

        print(f">>> RSI: {last_row['rsi']:.2f} (Price: {last_row['close']})")

        if last_row['rsi'] < 30:
            return "BUY_SIGNAL", 0.8 # 訊號, 信心度
        elif last_row['rsi'] > 70:
            return "SELL_SIGNAL", 0.8
        else:
            return "HOLD", 0.0

# ==========================================
# 4. 機器學習策略 (Random Forest)
# ==========================================
class MLStrategy(BaseStrategy):
    def __init__(self, client, symbol=conf.SYMBOL, interval=conf.INTERVAL):
        # 顯式呼叫父類初始化，並傳遞參數
        super().__init__(client, symbol, interval)
        self.model = None
        self.features = ['RSI_14', 'SMA_10', 'SMA_50', 'CLOSE_PCT']

    def prepare_features(self, df):
        df = df.copy()
        df['RSI_14'] = df.ta.rsi(length=14)
        df['SMA_10'] = df.ta.sma(length=10)
        df['SMA_50'] = df.ta.sma(length=50)
        df['CLOSE_PCT'] = df['close'].pct_change()
        df.dropna(inplace=True)
        return df

    def train(self, df):
        print(">>> 訓練 Random Forest 模型...")
        df = self.prepare_features(df)
        df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
        df.dropna(inplace=True)

        X = df[self.features]
        y = df['Target']

        # 簡單切分
        split = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:split], y.iloc[:split]

        self.model = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
        self.model.fit(X_train, y_train)
        print(">>> 模型訓練完成")

    async def analyze(self):
        # 1. 獲取足夠數據來訓練 + 預測
        df = await self.fetch_data(limit=1000)

        # 2. 如果沒模型就先訓練
        if self.model is None:
            self.train(df)

        # 3. 準備最新數據進行預測
        df_processed = self.prepare_features(df)
        last_row = df_processed.iloc[[-1]][self.features]

        prediction = self.model.predict(last_row)[0]
        probs = self.model.predict_proba(last_row)[0]
        confidence = probs[prediction]

        signal = "BUY_SIGNAL" if prediction == 1 else "SELL_SIGNAL"
        return signal, confidence

# ==========================================
# 5. 深度學習策略 (LSTM)
# ==========================================
class DeepStrategy(BaseStrategy):
    # [FIX 2] 修正 __init__，必須接收 symbol 和 interval 並傳給父類
    def __init__(self, client, symbol=conf.SYMBOL, interval=conf.INTERVAL):
        super().__init__(client, symbol, interval)
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.seq_length = 60
        self.hidden_size = 64

    def create_sequences(self, data):
        xs, ys = [], []
        for i in range(len(data) - self.seq_length):
            x = data[i:(i + self.seq_length)]
            y = data[i + self.seq_length]
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def train(self, df):
        print(f">>> 訓練 LSTM 模型 (Device: {conf.DEVICE})...")
        data = df[['close']].values
        data_scaled = self.scaler.fit_transform(data)

        X, y = self.create_sequences(data_scaled)

        X_tensor = torch.from_numpy(X).float().to(conf.DEVICE)
        y_tensor = torch.from_numpy(y).float().to(conf.DEVICE)

        # 簡單訓練迴圈 (不使用 DataLoader 以簡化範例)
        self.model = CryptoLSTM(1, self.hidden_size).to(conf.DEVICE)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

        self.model.train()
        for epoch in range(20): # 20 Epochs
            optimizer.zero_grad()
            output = self.model(X_tensor) # 注意：這裡維度要對齊
            output = output.squeeze() # (N, 1) -> (N)
            loss = criterion(output, y_tensor.squeeze())
            loss.backward()
            optimizer.step()

        print(f">>> LSTM 訓練完成 (Loss: {loss.item():.4f})")

    async def analyze(self):
        df = await self.fetch_data(limit=1000)

        if self.model is None:
            self.train(df)

        self.model.eval()
        last_seq = df[['close']].values[-self.seq_length:]
        last_seq_scaled = self.scaler.transform(last_seq)

        X_input = torch.from_numpy(last_seq_scaled).float().unsqueeze(0).to(conf.DEVICE) # (1, 60, 1)

        with torch.no_grad():
            pred_scaled = self.model(X_input).cpu().numpy()

        pred_price = self.scaler.inverse_transform(pred_scaled)[0][0]
        current_price = df.iloc[-1]['close']

        print(f">>> LSTM 預測: {pred_price:.2f} (現價: {current_price})")

        if pred_price > current_price * 1.001:
            return "BUY_SIGNAL", (pred_price - current_price)/current_price
        elif pred_price < current_price * 0.999:
            return "SELL_SIGNAL", (current_price - pred_price)/current_price
        else:
            return "HOLD", 0.0

# ==========================================
# 6. 統一入口工廠
# ==========================================
# [FIX 3] 修正縮排：這個函式應該在最外層，不能縮在 class 裡面
async def run_analysis_module(mode: str = 'rsi', symbol: str = 'BTCUSDT', interval: str = '15m'):
    """
    現在這個函數接收 symbol 和 interval 了！
    """
    conf.validate_api_config()
    logger = get_logger("bat.analyzer")
    client = create_spot_client()

    try:
        strategy = None
        # 初始化時將 symbol, interval 傳進去
        if mode == 'rsi':
            strategy = RSIStrategy(client, symbol, interval)
        elif mode == 'ml':
            strategy = MLStrategy(client, symbol, interval)
        elif mode == 'lstm':
            strategy = DeepStrategy(client, symbol, interval)
        else:
            return "UNKNOWN", 0.0

        print(f"=== 執行分析: {mode.upper()} | {symbol} | {interval} ===")
        logger.info("Run analysis: mode=%s symbol=%s interval=%s", mode, symbol, interval)
        try:
            return await strategy.analyze()
        except Exception:
            logger.exception("Strategy failed, fallback to ERROR result")
            return "ERROR", 0.0

    except Exception as e:
        print(f"錯誤: {e}")
        logger.exception("Analysis failed")
        return "ERROR", 0.0
    finally:
        client = None

if __name__ == "__main__":
    # 測試用：可以手動改這裡來測不同模式
    asyncio.run(run_analysis_module('lstm'))
