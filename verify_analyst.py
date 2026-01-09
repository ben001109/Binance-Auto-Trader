# Test Script
import asyncio
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from bat.analyzer import AnalystAgent, TradeDecision, RiskParams
from bat.config import conf

# Mock asyncio loop
async def main():
    print(">>> Testing AnalystAgent integration...")
    
    # 1. Test Instantiation
    agent = AnalystAgent(client=None, mode='lstm', symbol='BTCUSDT', interval='15m')
    # Point to small mock data to avoid loading 500MB history.csv during verification
    agent.strategy.training_data_path = "data/history_mock.csv"
    print("Agent created.")
    
    # 2. Test Analyze (with Mock data)
    # Mock Klines (list of [timestamp, open, high, low, close, volume, ...])
    # Need enough data for LSTM (60+)
    klines = []
    base_price = 50000.0
    for i in range(100):
        klines.append([
            1600000000000 + i*60000,
            str(base_price), str(base_price+100), str(base_price-100), str(base_price+10), "1.0",
            1600000000000 + i*60000 + 59999, "50000.0", 10, "1.0", "50000.0", "0"
        ])
        base_price += 10 # Slight uptrend
        
    print(f"Mocking {len(klines)} klines...")
    decision, risk = await agent.analyze(klines)
    
    print("Analysis Result:")
    print(f"Action: {decision.action}")
    print(f"Confidence: {decision.confidence}")
    print(f"Source: {decision.source}")
    if risk:
        print(f"Risk: SL={risk.stop_loss} TP={risk.take_profit}")
        
    # 3. Test Risk Profile Logic
    print("\nTesting Risk Profiles...")
    profiles = ["CONSERVATIVE", "STANDARD", "AGGRESSIVE"]
    for p in profiles:
        conf.RISK_PROFILE = p
        d, r = await agent.analyze(klines)
        if r:
            print(f"[{p}] SL={r.stop_loss:.4f} TP={r.take_profit:.4f}")

if __name__ == "__main__":
    asyncio.run(main())
