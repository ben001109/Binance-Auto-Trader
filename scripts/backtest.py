import argparse
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from bat.backtest import run_backtest
from bat.logger import install_crash_handler


def main():
    install_crash_handler()
    parser = argparse.ArgumentParser(description="BAT Backtest (RSI baseline)")
    parser.add_argument("--data", default="data/history.csv")
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--slippage", type=float, default=0.0005)
    parser.add_argument("--splits", type=int, default=3)
    parser.add_argument("--stop-loss", type=float, default=None)
    parser.add_argument("--take-profit", type=float, default=None)
    parser.add_argument("--max-dd-stop", type=float, default=None)
    args = parser.parse_args()

    result = run_backtest(
        data_path=args.data,
        fee=args.fee,
        slippage=args.slippage,
        position_splits=args.splits,
        stop_loss_pct=args.stop_loss,
        take_profit_pct=args.take_profit,
        max_drawdown_stop=args.max_dd_stop,
    )
    print("=== Backtest Result (RSI) ===")
    print(f"Total Return: {result.total_return:.2%}")
    print(f"Max Drawdown: {result.max_drawdown:.2%}")
    print(f"Trades: {result.trades}")
    print(f"Win Rate: {result.win_rate:.2%}")
    print(f"Loss Rate: {result.loss_rate:.2%}")
    print(f"Avg Win: {result.avg_win:.2%}")
    print(f"Avg Loss: {result.avg_loss:.2%}")


if __name__ == "__main__":
    main()
