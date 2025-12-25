import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from bat.training import train_and_backtest
from bat.logger import install_crash_handler


def train():
    install_crash_handler()
    result, risk = train_and_backtest()
    print("=== Backtest Result (RSI) ===")
    print(f"Total Return: {result.total_return:.2%}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"Trades: {result.trades}")
    print(f"Win Rate: {result.win_rate:.2%}")
    print(f"Loss Rate: {result.loss_rate:.2%}")
    print(f"Avg Win: {result.avg_win:.2%}")
    print(f"Avg Loss: {result.avg_loss:.2%}")
    print("=== Suggested Risk Params ===")
    print(f"Stop Loss: {risk.stop_loss:.2%}")
    print(f"Take Profit: {risk.take_profit:.2%}")
    print(f"Max DD Stop: {risk.max_dd_stop:.2%}")
    print(f"Position Splits: {risk.position_splits}")


if __name__ == "__main__":
    train()
