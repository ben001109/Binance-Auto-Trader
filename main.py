import sys
import os
import argparse
import asyncio

# ----------------------------------------------------------------
# 1. 路徑設定 (重要！)
# 確保 Python 能找到 src 資料夾下的 bat 套件
# 這樣你就不用每次都打 export PYTHONPATH=...
# ----------------------------------------------------------------
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from bat.config import conf
from bat.logger import install_crash_handler

def main():
    install_crash_handler()
    # 2. 參數解析 (Argument Parsing)
    # 讓你可以用指令控制要開啟 TUI 還是直接跑某個策略
    parser = argparse.ArgumentParser(description="Binance Auto Trader (BAT) - Starforge Edition")

    parser.add_argument(
        '--mode',
        type=str,
        default='tui',
        choices=['tui', 'rsi', 'ml', 'lstm', 'verify'],
        help="選擇運行模式: tui (圖形介面, 預設), rsi, ml, lstm (純文字分析模式), verify (系統驗證)"
    )

    args = parser.parse_args()

    # 3. 顯示啟動資訊
    print("=" * 40)
    print(f"🚀 BAT 系統啟動中...")
    print(f"🔧 運行模式: {args.mode.upper()}")
    print(f"🌍 環境設定: {'TESTNET (測試網)' if conf.IS_TESTNET else 'REAL MONEY (實盤 WARNING)'}")
    print("=" * 40)

    # Windows Asyncio Fix
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    try:
        # 4. 模式分發 (Dispatcher)
        if args.mode == 'tui':
            # 啟動圖形介面
            from bat.tui.app import CryptoApp
            app = CryptoApp()
            app.run()
        elif args.mode == "cli": # New CLI mode
            from bat.cli.app import run_cli # Assuming run_cli is in bat.cli.app
            asyncio.run(run_cli())
        elif args.mode == 'verify':
            # 系統驗證模式
            from bat.verifier import Verifier
            asyncio.run(Verifier.run_verification())
        else:
            # 啟動 CLI 分析模式 (RSI / ML / LSTM)
            from bat.analyzer import run_analysis_module

            # 因為 run_analysis_module 是 async 函式，需要用 asyncio.run 執行
            asyncio.run(run_analysis_module(mode=args.mode))

    except KeyboardInterrupt:
        print("\n👋 使用者手動中斷程式，再見！")
    except Exception as e:
        import traceback
        print(f"\n❌ 發生未預期的錯誤: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
