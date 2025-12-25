from dataclasses import dataclass

import pandas as pd
import pandas_ta as ta

from bat.logger import get_logger


@dataclass
class BacktestResult:
    total_return: float
    max_drawdown: float
    trades: int
    win_rate: float
    loss_rate: float
    avg_win: float
    avg_loss: float
    equity_curve: list[float] | None = None


def load_history_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    for col in ["open", "high", "low", "close", "volume"]:
        if col in df.columns:
            df[col] = df[col].astype(float)
    return df


def generate_rsi_signals(
    df: pd.DataFrame,
    length: int = 14,
    lower: float = 30,
    upper: float = 70,
) -> pd.Series:
    rsi = df.ta.rsi(length=length)
    signals = pd.Series("HOLD", index=df.index)
    signals[rsi < lower] = "BUY"
    signals[rsi > upper] = "SELL"
    return signals


def backtest_signals(
    df: pd.DataFrame,
    signals: pd.Series,
    initial_cash: float = 1000.0,
    fee: float = 0.001,
    slippage: float = 0.0005,
    position_splits: int = 3,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    max_drawdown_stop: float | None = None,
    return_equity: bool = False,
    equity_points: int = 200,
) -> BacktestResult:
    cash = initial_cash
    trade_returns = []
    equity_curve = []
    lots = []
    peak_equity = initial_cash
    trading_halted = False

    for idx, row in df.iterrows():
        price = float(row["close"])
        signal = signals.iloc[idx]

        if lots:
            remaining_lots = []
            for lot in lots:
                stop_trigger = stop_loss_pct is not None and price <= lot["price"] * (1 - stop_loss_pct)
                take_trigger = take_profit_pct is not None and price >= lot["price"] * (1 + take_profit_pct)
                if stop_trigger or take_trigger:
                    exec_price = price * (1 - slippage)
                    proceeds = lot["qty"] * exec_price * (1 - fee)
                    cash += proceeds
                    trade_returns.append((exec_price - lot["price"]) / lot["price"])
                else:
                    remaining_lots.append(lot)
            lots = remaining_lots

        if not trading_halted:
            if signal == "BUY" and len(lots) < position_splits and cash > 0:
                tranche_pct = 1.0 / position_splits
                equity = cash + sum(lot["qty"] for lot in lots) * price
                invest = min(cash, equity * tranche_pct)
                if invest > 0:
                    exec_price = price * (1 + slippage)
                    qty = (invest * (1 - fee)) / exec_price
                    cash -= invest
                    lots.append({"qty": qty, "price": exec_price})
            elif signal == "SELL" and lots:
                lot = lots.pop(0)
                exec_price = price * (1 - slippage)
                proceeds = lot["qty"] * exec_price * (1 - fee)
                cash += proceeds
                trade_returns.append((exec_price - lot["price"]) / lot["price"])

        position_value = sum(lot["qty"] for lot in lots) * price
        equity = cash + position_value
        equity_curve.append(equity)
        peak_equity = max(peak_equity, equity)

        if max_drawdown_stop is not None and peak_equity > 0:
            drawdown = (equity - peak_equity) / peak_equity
            if drawdown <= -max_drawdown_stop:
                trading_halted = True

    if lots:
        last_price = float(df.iloc[-1]["close"])
        exec_price = last_price * (1 - slippage)
        for lot in lots:
            proceeds = lot["qty"] * exec_price * (1 - fee)
            cash += proceeds
            trade_returns.append((exec_price - lot["price"]) / lot["price"])
        lots = []

    equity_series = pd.Series(equity_curve)
    running_max = equity_series.cummax()
    drawdown = (equity_series - running_max) / running_max
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0

    trades = len(trade_returns)
    wins = [r for r in trade_returns if r > 0]
    losses = [r for r in trade_returns if r <= 0]
    win_rate = (len(wins) / trades) if trades else 0.0
    loss_rate = (len(losses) / trades) if trades else 0.0

    total_return = (cash - initial_cash) / initial_cash
    avg_win = sum(wins) / len(wins) if wins else 0.0
    avg_loss = sum(losses) / len(losses) if losses else 0.0

    return BacktestResult(
        total_return=total_return,
        max_drawdown=max_drawdown,
        trades=trades,
        win_rate=win_rate,
        loss_rate=loss_rate,
        avg_win=avg_win,
        avg_loss=avg_loss,
        equity_curve=equity_curve[-equity_points:] if return_equity else None,
    )


def run_backtest(
    data_path: str = "data/history.csv",
    strategy: str = "rsi",
    initial_cash: float = 1000.0,
    fee: float = 0.001,
    slippage: float = 0.0005,
    position_splits: int = 3,
    stop_loss_pct: float | None = None,
    take_profit_pct: float | None = None,
    max_drawdown_stop: float | None = None,
    return_equity: bool = False,
    equity_points: int = 200,
) -> BacktestResult:
    df = load_history_csv(data_path)
    logger = get_logger("bat.backtest")
    logger.info(
        "Backtest start: strategy=%s data=%s fee=%.4f slippage=%.4f splits=%s",
        strategy,
        data_path,
        fee,
        slippage,
        position_splits,
    )

    if strategy == "rsi":
        signals = generate_rsi_signals(df)
    else:
        raise ValueError(f"Unsupported strategy: {strategy}")

    result = backtest_signals(
        df,
        signals,
        initial_cash=initial_cash,
        fee=fee,
        slippage=slippage,
        position_splits=position_splits,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        max_drawdown_stop=max_drawdown_stop,
        return_equity=return_equity,
        equity_points=equity_points,
    )
    logger.info(
        "Backtest end: return=%.4f max_dd=%.4f trades=%s win=%.4f loss=%.4f",
        result.total_return,
        result.max_drawdown,
        result.trades,
        result.win_rate,
        result.loss_rate,
    )
    return result
