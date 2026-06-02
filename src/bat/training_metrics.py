from __future__ import annotations

import numpy as np


def distribution(values) -> dict[int, int]:
    array = np.asarray(values).astype(int).reshape(-1)
    return {label: int((array == label).sum()) for label in (0, 1, 2)}


def _f1_for_label(y_true: np.ndarray, y_pred: np.ndarray, label: int) -> float:
    true_positive = float(((y_true == label) & (y_pred == label)).sum())
    false_positive = float(((y_true != label) & (y_pred == label)).sum())
    false_negative = float(((y_true == label) & (y_pred != label)).sum())
    denominator = (2 * true_positive) + false_positive + false_negative
    if denominator == 0:
        return 0.0
    return (2 * true_positive) / denominator


def classification_report(y_true, y_pred) -> dict:
    true = np.asarray(y_true).astype(int).reshape(-1)
    pred = np.asarray(y_pred).astype(int).reshape(-1)
    if len(true) != len(pred):
        raise ValueError("y_true and y_pred must have the same length")
    accuracy = float((true == pred).mean()) if len(true) else 0.0
    macro_f1 = float(np.mean([_f1_for_label(true, pred, label) for label in (0, 1, 2)]))
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "label_distribution": distribution(true),
        "prediction_distribution": distribution(pred),
    }


def max_drawdown(equity_curve) -> float:
    equity = np.asarray(equity_curve, dtype=float).reshape(-1)
    if len(equity) == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    valid = running_max > 0
    drawdowns = np.zeros_like(equity, dtype=float)
    drawdowns[valid] = (equity[valid] - running_max[valid]) / running_max[valid]
    return float(drawdowns.min())


def _trade_returns_after_cost(predictions, future_returns, fee: float, slippage: float) -> np.ndarray:
    pred = np.asarray(predictions).astype(int).reshape(-1)
    returns = np.asarray(future_returns, dtype=float).reshape(-1)
    if len(pred) != len(returns):
        raise ValueError("predictions and future_returns must have the same length")
    cost = float(fee) + float(slippage)
    trade_returns = []
    for signal, future_return in zip(pred, returns, strict=True):
        if signal == 2:
            trade_returns.append(float(future_return) - cost)
        elif signal == 0:
            trade_returns.append(-float(future_return) - cost)
    return np.asarray(trade_returns, dtype=float)


def trading_report(predictions, future_returns, fee: float, slippage: float) -> dict:
    trade_returns = _trade_returns_after_cost(predictions, future_returns, fee, slippage)
    trade_count = int(len(trade_returns))
    if trade_count == 0:
        return {
            "trade_count": 0,
            "expected_return_after_cost": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "max_drawdown": 0.0,
        }

    wins = trade_returns[trade_returns > 0]
    losses = trade_returns[trade_returns < 0]
    gross_profit = float(wins.sum())
    gross_loss = float(abs(losses.sum()))
    equity = 100.0 * np.cumprod(1.0 + trade_returns)
    return {
        "trade_count": trade_count,
        "expected_return_after_cost": float(trade_returns.mean()),
        "win_rate": float(len(wins) / trade_count),
        "profit_factor": gross_profit / gross_loss if gross_loss > 0 else float("inf"),
        "max_drawdown": max_drawdown(equity),
    }
