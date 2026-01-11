import asyncio
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pandas_ta as ta
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier

from bat.config import conf
from bat.execution.spot_client import async_klines, create_spot_client
from bat.logger import get_logger
from bat.models.lstm import CryptoLSTM
from bat.data.dataset import DataProcessor
from bat.training import RiskParams, suggest_risk_params_from_model, _load_training_data
from bat.data.integrity import check_data_gaps, heal_data_gaps, merge_healed_data

# ==========================================
# Data Structures
# ==========================================

@dataclass
class TradeDecision:
    action: str          # BUY, SELL, HOLD
    confidence: float    # 0.0 - 1.0
    current_price: float
    predicted_price: float
    invest_amount: float # Quote asset amount to use (0.0 if not decided here)
    source: str          # model, rsi, ml, fallback, filtered
    signal: str          # Original signal before filtering

# ==========================================
# Base Strategy
# ==========================================

class BaseStrategy:
    def __init__(self, client, symbol=None, interval=None):
        self.client = client
        self.symbol = symbol or conf.SYMBOL
        self.interval = interval or conf.INTERVAL
        self.logger = get_logger(f"bat.analyst.{self.__class__.__name__}")

    async def fetch_data(self, limit=100):
        # self.logger.debug("Fetch data: %s %s limit=%s", self.symbol, self.interval, limit)
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

    async def analyze(self, klines=None) -> tuple[TradeDecision, RiskParams | None]:
        raise NotImplementedError("Subclasses must implement analyze")

# ==========================================
# Strategies
# ==========================================

class RSIStrategy(BaseStrategy):
    async def analyze(self, klines=None):
        if klines is None:
            df = await self.fetch_data(limit=100)
        else:
            df = klines if isinstance(klines, pd.DataFrame) else pd.DataFrame(klines) # Handle raw list if passed

        # Ensure numeric
        cols = ['close']
        for c in cols:
            if c in df.columns:
                df[c] = df[c].astype(float)

        df['rsi'] = df.ta.rsi(length=14)
        last_row = df.iloc[-1]
        current_price = float(last_row['close'])
        
        rsi_val = last_row['rsi']
        self.logger.info(f"RSI: {rsi_val:.2f} (Price: {current_price})")

        signal = "HOLD"
        confidence = 0.0
        
        if rsi_val < 30:
            signal = "BUY"
            confidence = (30 - rsi_val) / 30.0 # Simple linear confidence
        elif rsi_val > 70:
            signal = "SELL"
            confidence = (rsi_val - 70) / 30.0

        decision = TradeDecision(
            action=signal,
            confidence=min(confidence + 0.5, 1.0) if signal != "HOLD" else 0.0,
            current_price=current_price,
            predicted_price=current_price, # RSI doesn't predict price
            invest_amount=0.0,
            source="rsi",
            signal=signal
        )
        return decision, None # No specific risk params for RSI yet

class MLStrategy(BaseStrategy):
    def __init__(self, client, symbol=None, interval=None):
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
        self.logger.info("Training Random Forest model...")
        df = self.prepare_features(df)
        df['Target'] = (df['close'].shift(-1) > df['close']).astype(int)
        df.dropna(inplace=True)

        X = df[self.features]
        y = df['Target']

        if len(X) < 50:
             self.logger.warning("Not enough data to train RF")
             return

        split = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:split], y.iloc[:split]

        self.model = RandomForestClassifier(n_estimators=100, min_samples_split=10, random_state=42)
        self.model.fit(X_train, y_train)
        self.logger.info("Model training complete")

    async def analyze(self, klines=None):
        if klines is None:
            df = await self.fetch_data(limit=1000)
        else:
            df = klines if isinstance(klines, pd.DataFrame) else pd.DataFrame(klines)

        cols = ['close']
        for c in cols:
            if c in df.columns:
                 df[c] = df[c].astype(float)

        if self.model is None:
            self.train(df)

        if self.model is None:
             return TradeDecision("HOLD", 0.0, 0.0, 0.0, 0.0, "ml_error", "HOLD"), None

        df_processed = self.prepare_features(df)
        if df_processed.empty:
             return TradeDecision("HOLD", 0.0, 0.0, 0.0, 0.0, "ml_nodata", "HOLD"), None

        last_row = df_processed.iloc[[-1]][self.features]
        prediction = self.model.predict(last_row)[0]
        probs = self.model.predict_proba(last_row)[0]
        confidence = probs[prediction]
        current_price = float(df.iloc[-1]['close'])

        signal = "BUY" if prediction == 1 else "SELL"
        
        decision = TradeDecision(
            action=signal,
            confidence=float(confidence),
            current_price=current_price,
            predicted_price=current_price,
            invest_amount=0.0,
            source="ml",
            signal=signal
        )
        return decision, None

class LSTMStrategy(BaseStrategy):
    def __init__(self, client, symbol=None, interval=None):
        super().__init__(client, symbol, interval)
        self.model_path = os.path.join("data", "lstm_model.pth")
        self.training_data_path = os.path.join("data", "history.csv")
        self._missing_model_warned = False
        self.processor = None
        self.train_df = None

    def _prepare_market_df(self, klines) -> pd.DataFrame:
        if isinstance(klines, pd.DataFrame):
            df = klines.copy()
        else:
            df = pd.DataFrame(klines, columns=[
                "timestamp", "open", "high", "low", "close", "volume",
                "close_time", "q_vol", "trades", "tb_base", "tb_quote", "ignore"
            ])
        cols = ["open", "high", "low", "close", "volume"]
        for c in cols:
            if c in df.columns:
                df[c] = df[c].astype(float)
        return df

    def _load_model(self) -> CryptoLSTM | None:
        if not os.path.exists(self.model_path):
            return None
        model = CryptoLSTM(
            input_dim=len(conf.FEATURE_COLS),
            hidden_dim=conf.HIDDEN_SIZE,
            num_layers=conf.NUM_LAYERS,
            dropout=conf.DROPOUT
        ).to(conf.DEVICE)
        try:
            state = torch.load(self.model_path, map_location=conf.DEVICE)
            model.load_state_dict(state)
        except Exception as exc:
            self.logger.warning(f"Model load failed ({self.model_path}): {exc}")
            return None
        model.eval()
        return model

    def _predict_probs(self, model: CryptoLSTM, processor: DataProcessor, market_df: pd.DataFrame) -> np.ndarray:
        data_scaled, _ = processor.process_for_inference(market_df, conf.FEATURE_COLS)
        if len(data_scaled) < conf.SEQ_LENGTH:
            raise ValueError("Insufficient data for prediction")
        last_seq = data_scaled[-conf.SEQ_LENGTH:]
        x = torch.FloatTensor(last_seq).unsqueeze(0).to(conf.DEVICE)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        return probs

    def _fallback_decision(self, market_df: pd.DataFrame) -> tuple[TradeDecision, RiskParams | None]:
        closes = market_df["close"].astype(float)
        returns = closes.pct_change().dropna()
        if returns.empty:
            return TradeDecision("HOLD", 0.0, 0.0, 0.0, 0.0, "fallback_nodata", "HOLD"), None
        
        last_return = float(returns.iloc[-1])
        vol = float(returns.std()) if len(returns) > 1 else 0.0
        threshold = max(0.001, vol * 1.5)
        
        if last_return > threshold:
            action = "BUY"
        elif last_return < -threshold:
            action = "SELL"
        else:
            action = "HOLD"
            
        confidence = 0.0
        if vol > 0:
            confidence = min(abs(last_return) / (vol * 3.0), 1.0)
            
        current_price = float(closes.iloc[-1])
        predicted_price = current_price * (1.0 + last_return)
        
        risk = RiskParams(
            stop_loss=min(max(vol * 2.0, 0.01), 0.12),
            take_profit=min(max(vol * 3.0, 0.02), 0.2),
            max_dd_stop=min(max(vol * 6.0, 0.05), 0.3),
            position_splits=3,
        )
        
        decision = TradeDecision(
            action=action,
            confidence=confidence,
            current_price=current_price,
            predicted_price=predicted_price,
            invest_amount=0.0,
            source="fallback",
            signal=action,
        )
        return decision, risk

    async def analyze(self, klines=None) -> tuple[TradeDecision, RiskParams | None]:
        if klines is None:
            # If no klines provided, fetch them (usually for CLI usage)
            market_df = await self.fetch_data(limit=conf.SEQ_LENGTH + 50)
        else:
            market_df = self._prepare_market_df(klines)

        if len(market_df) < conf.SEQ_LENGTH + 1:
            self.logger.warning("Insufficient market data for prediction")
            return TradeDecision("HOLD", 0.0, 0.0, 0.0, 0.0, "insufficient_data", "HOLD"), None

        model = self._load_model()
        if model is None:
            if not self._missing_model_warned:
                self.logger.warning(f"Model not found: {self.model_path}, using fallback")
                self._missing_model_warned = True
            return self._fallback_decision(market_df)

        if self.processor is None:
            self.processor = DataProcessor()
            # Try fit processor on training data for better scaling, else use market data
            try:
                if self.train_df is None:
                    self.train_df = _load_training_data(self.training_data_path)
                self.processor.process_for_training(self.train_df, conf.FEATURE_COLS)
            except Exception:
                self.logger.warning("Training data unavailable (or load failed), fitting processor on market data only")
                self.train_df = market_df
                self.processor.process_for_training(self.train_df, conf.FEATURE_COLS)

        try:
            probs = self._predict_probs(model, self.processor, market_df)
        except ValueError:
             return self._fallback_decision(market_df)

        expected_return = (probs[2] - probs[0]) * conf.RETURN_THRESHOLD
        current_price = float(market_df.iloc[-1]["close"])
        predicted_price = current_price * (1.0 + expected_return)

        # Risk Calculation
        try:
            model_risk = suggest_risk_params_from_model(model, self.processor, self.train_df)
        except Exception:
            self.logger.exception("Failed to derive risk from model, using default")
            model_risk = RiskParams(stop_loss=0.02, take_profit=0.05, max_dd_stop=0.2, position_splits=3)
        
        market_vol = float(market_df["close"].pct_change().std())
        risk = self._blend_risk_params(model_risk, market_vol)

        # Signal Logic
        signal_idx = int(np.argmax(probs))
        signal_map = {0: "SELL", 1: "HOLD", 2: "BUY"}
        signal = signal_map.get(signal_idx, "HOLD")
        confidence = float(np.max(probs))

        decision = TradeDecision(
            action=signal,
            confidence=confidence,
            current_price=current_price,
            predicted_price=predicted_price,
            invest_amount=0.0,
            source="model",
            signal=signal
        )
        return decision, risk

    def _blend_risk_params(self, model_params: RiskParams, market_vol: float) -> RiskParams:
        # Risk Multipliers based on RISK_PROFILE
        profile = getattr(conf, 'RISK_PROFILE', 'STANDARD')
        
        if profile == 'CONSERVATIVE':
            sl_mult, tp_mult, dd_mult = 1.5, 2.0, 4.0
        elif profile == 'AGGRESSIVE':
            sl_mult, tp_mult, dd_mult = 3.0, 5.0, 8.0
        else: # STANDARD
            sl_mult, tp_mult, dd_mult = 2.0, 3.0, 6.0

        stop_loss = min(max(max(model_params.stop_loss, market_vol * sl_mult), 0.01), 0.15)
        take_profit = min(max(max(model_params.take_profit, market_vol * tp_mult), 0.02), 0.25)
        max_dd_stop = min(max(max(model_params.max_dd_stop, market_vol * dd_mult), 0.05), 0.4)
        
        return RiskParams(
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_dd_stop=max_dd_stop,
            position_splits=model_params.position_splits,
        )

# ==========================================
# Analyst Agent Factory
# ==========================================

class AnalystAgent:
    def __init__(self, client=None, mode='lstm', symbol=None, interval=None):
        self.client = client or create_spot_client(is_testnet=conf.IS_TESTNET)
        self.mode = mode
        self.symbol = symbol or conf.SYMBOL
        self.interval = interval or conf.INTERVAL
        self.logger = get_logger("bat.analyst")
        
        if mode == 'rsi':
            self.strategy = RSIStrategy(self.client, self.symbol, self.interval)
        elif mode == 'ml':
            self.strategy = MLStrategy(self.client, self.symbol, self.interval)
        elif mode == 'lstm':
            self.strategy = LSTMStrategy(self.client, self.symbol, self.interval)
        else:
             self.strategy = LSTMStrategy(self.client, self.symbol, self.interval) # Default

    async def analyze(self, klines=None) -> tuple[TradeDecision, RiskParams | None]:
        try:
             return await self.strategy.analyze(klines)
        except Exception as e:
             self.logger.exception(f"Analysis Failed in {self.mode}")
             # Return error decision
             return TradeDecision("HOLD", 0.0, 0.0, 0.0, 0.0, "error", "HOLD"), None

    async def ensure_data_integrity(self, on_status=None):
        """
        Check and heal gaps in historical data.
        Runs blocking I/O in a separate thread.
        """
        if self.mode != 'lstm':
            return
            
        msg = "Checking data integrity..."
        self.logger.info(msg)
        if on_status: on_status(msg)

        strategy = self.strategy
        
        if hasattr(strategy, 'training_data_path') and os.path.exists(strategy.training_data_path):
             # Define the sync blocking function
            def _check_and_heal_sync():
                try:
                    df = pd.read_csv(strategy.training_data_path)
                    interval_ms = 15 * 60 * 1000 
                    unit = self.interval[-1]
                    val = int(self.interval[:-1])
                    if unit == 'm': interval_ms = val * 60 * 1000
                    elif unit == 'h': interval_ms = val * 60 * 60 * 1000
                    elif unit == 'd': interval_ms = val * 24 * 60 * 60 * 1000
                    
                    gaps = check_data_gaps(df, interval_ms)
                    return df, gaps
                except Exception as e:
                    self.logger.error(f"Integrity check error: {e}")
                    return None, []

            # Run read/check in thread
            if on_status: on_status("正在讀取並檢查歷史資料 (可能需要幾秒鐘)...")
            df, gaps = await asyncio.to_thread(_check_and_heal_sync)
            
            if gaps:
                msg = f"發現 {len(gaps)} 個資料缺口，正在修補..."
                self.logger.warning(msg)
                if on_status: on_status(msg)

                # Healing is async IO, can run on main loop
                new_chunks = await heal_data_gaps(self.client, self.symbol, self.interval, gaps)
                
                if new_chunks:
                    msg = f"下載了 {len(new_chunks)} 筆新資料，正在寫入..."
                    if on_status: on_status(msg)

                    # merging and saving is blocking again
                    def _save_healed():
                        df_healed = merge_healed_data(df, new_chunks)
                        df_healed.to_csv(strategy.training_data_path, index=False)
                    
                    await asyncio.to_thread(_save_healed)
                    self.logger.info(f"Healed data saved to {strategy.training_data_path}")
                    if on_status: on_status("資料修補完成並已存檔。")
                else:
                    self.logger.warning("No data found to heal gaps.")
                    if on_status: on_status("警告: 無法下載缺口資料。")
            else:
                self.logger.info("Data Integrity Check Passed: No gaps found.")
                if on_status: on_status("資料完整性檢查通過。")

# ==========================================
# Legacy Helper (for Backwards Compatibility if needed)
# ==========================================

async def run_analysis_module(mode: str = 'rsi', symbol: str = 'BTCUSDT', interval: str = '15m'):
    agent = AnalystAgent(mode=mode, symbol=symbol, interval=interval)
    decision, risk = await agent.analyze()
    print(f"Decision: {decision.action} | Conf: {decision.confidence:.2f} | Source: {decision.source}")
    return decision.action, decision.confidence

# ==========================================
# Legacy / Utility Helpers (Moved from auto_trader.py)
# ==========================================

def compute_order_size(quote_balance: float, market_vol: float) -> float:
    fraction = min(max(market_vol * 5.0, 0.02), 0.2)
    return quote_balance * fraction

def apply_confidence_threshold(decision: TradeDecision | None, threshold: float) -> TradeDecision | None:
    if decision is None:
        return None
    if decision.confidence < threshold:
        decision.action = "HOLD"
        decision.source = "filtered"
    else:
        decision.action = decision.signal
    return decision
