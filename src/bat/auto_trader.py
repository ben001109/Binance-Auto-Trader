import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from bat.config import conf
from bat.data.dataset import DataProcessor
from bat.logger import get_logger
from bat.models.lstm import CryptoLSTM
from bat.training import RiskParams, suggest_risk_params_from_model, _load_training_data

_missing_model_warned = False


@dataclass
class TradeDecision:
    action: str
    confidence: float
    current_price: float
    predicted_price: float
    invest_amount: float
    source: str
    signal: str


def _symbol_assets(symbol: str):
    if symbol.endswith("USDT"):
        return symbol[:-4], "USDT"
    return symbol[:-3], symbol[-3:]


def _market_volatility(df: pd.DataFrame) -> float:
    returns = df["close"].pct_change().dropna()
    return float(returns.std()) if not returns.empty else 0.0


def _prepare_market_df(klines) -> pd.DataFrame:
    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "q_vol", "trades", "tb_base", "tb_quote", "ignore"
    ])
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df


def _load_model(model_path: str) -> CryptoLSTM | None:
    if not os.path.exists(model_path):
        return None
    model = CryptoLSTM(
        input_dim=len(conf.FEATURE_COLS),
        hidden_dim=conf.HIDDEN_SIZE,
        num_layers=conf.NUM_LAYERS,
        dropout=conf.DROPOUT
    ).to(conf.DEVICE)
    try:
        state = torch.load(model_path, map_location=conf.DEVICE)
        model.load_state_dict(state)
    except Exception as exc:
        logger = get_logger("bat.auto")
        logger.warning("Model load failed (%s), fallback to rule-based: %s", model_path, exc)
        return None
    model.eval()
    return model


def _predict_probs(model: CryptoLSTM, processor: DataProcessor, market_df: pd.DataFrame) -> np.ndarray:
    data_scaled, _ = processor.process_for_inference(market_df, conf.FEATURE_COLS)
    if len(data_scaled) < conf.SEQ_LENGTH:
        raise ValueError("Insufficient data for prediction")
    last_seq = data_scaled[-conf.SEQ_LENGTH:]
    x = torch.FloatTensor(last_seq).unsqueeze(0).to(conf.DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    return probs


def _blend_risk_params(
    model_params: RiskParams,
    market_vol: float,
) -> RiskParams:
    stop_loss = min(max(max(model_params.stop_loss, market_vol * 2.0), 0.01), 0.15)
    take_profit = min(max(max(model_params.take_profit, market_vol * 3.0), 0.02), 0.25)
    max_dd_stop = min(max(max(model_params.max_dd_stop, market_vol * 6.0), 0.05), 0.4)
    return RiskParams(
        stop_loss=stop_loss,
        take_profit=take_profit,
        max_dd_stop=max_dd_stop,
        position_splits=model_params.position_splits,
    )


def _position_size(balance: float, market_vol: float) -> float:
    fraction = min(max(market_vol * 5.0, 0.02), 0.2)
    return balance * fraction


def apply_confidence_threshold(decision: TradeDecision | None, threshold: float) -> TradeDecision | None:
    if decision is None:
        return None
    if decision.confidence < threshold:
        decision.action = "HOLD"
        decision.source = "filtered"
    else:
        decision.action = decision.signal
    return decision


def decide_trade(
    klines,
    model_path: str = "data/lstm_model.pth",
    training_data_path: str = "data/history.csv",
) -> tuple[TradeDecision | None, RiskParams | None]:
    logger = get_logger("bat.auto")
    market_df = _prepare_market_df(klines)
    if len(market_df) < conf.SEQ_LENGTH + 1:
        logger.warning("Insufficient market data for prediction")
        return None, None

    model = _load_model(model_path)
    if model is None:
        global _missing_model_warned
        if not _missing_model_warned:
            logger.warning("Model not found: %s", model_path)
            _missing_model_warned = True
        return _fallback_decision(market_df)

    processor = DataProcessor()
    try:
        train_df = _load_training_data(training_data_path)
        processor.process_for_training(train_df, conf.FEATURE_COLS)
    except Exception:
        logger.exception("Training data unavailable, fitting processor on market data only")
        train_df = market_df
        processor.process_for_training(train_df, conf.FEATURE_COLS)

    probs = _predict_probs(model, processor, market_df)
    expected_return = (probs[2] - probs[0]) * conf.RETURN_THRESHOLD
    current_price = float(market_df.iloc[-1]["close"])
    predicted_price = current_price * (1.0 + expected_return)
    market_vol = _market_volatility(market_df)

    try:
        model_risk = suggest_risk_params_from_model(model, processor, train_df)
    except Exception:
        logger.exception("Failed to derive risk params from training data, fallback to market only")
        model_risk = RiskParams(stop_loss=0.02, take_profit=0.05, max_dd_stop=0.2, position_splits=3)

    risk = _blend_risk_params(model_risk, market_vol)

    signal_idx = int(np.argmax(probs))
    if signal_idx == 2:
        signal = "BUY"
    elif signal_idx == 0:
        signal = "SELL"
    else:
        signal = "HOLD"
    confidence = float(np.max(probs))

    decision = TradeDecision(
        action=signal,
        confidence=confidence,
        current_price=current_price,
        predicted_price=predicted_price,
        invest_amount=0.0,
        source="model",
        signal=signal,
    )
    return decision, risk


def compute_order_size(quote_balance: float, market_vol: float) -> float:
    return _position_size(quote_balance, market_vol)
def _fallback_decision(market_df: pd.DataFrame) -> tuple[TradeDecision | None, RiskParams | None]:
    closes = market_df["close"].astype(float)
    returns = closes.pct_change().dropna()
    if returns.empty:
        return None, None
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
