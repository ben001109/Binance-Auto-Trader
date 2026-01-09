import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import asyncio
import pandas as pd
import numpy as np
import shutil
from datetime import datetime
from bat.data.dataset import DataProcessor
from bat.data.integrity import check_data_gaps
from bat.config import conf

def verify_data_integrity():
    print("\n[VERIFY] Data Integrity...")
    # Create DF with a gap
    # 0, 15, 30, (GAP 45), 60...
    base = datetime(2023, 1, 1, 0, 0, 0)
    timestamps = [base.timestamp() * 1000 + i * 15 * 60 * 1000 for i in range(10)]
    # Remove index 3 (45 min)
    timestamps.pop(3)
    
    df = pd.DataFrame({'timestamp': timestamps, 'close': np.random.randn(len(timestamps)) + 100})
    
    interval_ms = 15 * 60 * 1000
    gaps = check_data_gaps(df, interval_ms)
    print(f"Gaps found: {gaps}")
    
    # Expected gap: start at 30m + 15m = 45m. End at 60m - 15m = 45m. 
    # Logic in check_data_gaps: gap_start = ts[idx] + interval, gap_end = ts[idx+1] - interval
    # ts[2] is 30m. ts[3] is 60m. 
    # start = 30 + 15 = 45. end = 60 - 15 = 45.
    # gap is 45 to 45 (point gap?). Yes.
    
    if len(gaps) == 1 and gaps[0][0] == gaps[0][1]:
        print("✅ Correctly identified single gap.")
    else:
        print("❌ Gap detection failed or unexpected format.")

def verify_htf_features():
    print("\n[VERIFY] Multi-Timeframe Features...")
    # Create 4 hours of 15m data
    # 16 bars needed? 4 * 60 / 15 = 4 bar/h. 
    # 100 bars
    timestamps = [datetime(2023, 1, 1).timestamp() * 1000 + i * 15 * 60 * 1000 for i in range(100)]
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': 100, 'high': 105, 'low': 95, 
        'close': np.sin(np.linspace(0, 10, 100)) * 10 + 100,
        'volume': 1000
    })
    
    processor = DataProcessor()
    df_processed = processor.add_1h_features(df)
    
    if 'RSI_1H' in df_processed.columns and 'EMA_20_1H' in df_processed.columns:
        print("✅ HTF Columns present.")
        print(f"Sample RSI_1H: {df_processed['RSI_1H'].iloc[-5:].values}")
    else:
        print("❌ HTF Columns missing.")

def verify_cost_aware_targets():
    print("\n[VERIFY] Cost-Aware Targets...")
    processor = DataProcessor()
    
    # Mock data
    timestamps = [datetime(2023, 1, 1).timestamp() * 1000 + i * 15 * 60 * 1000 for i in range(200)]
    df = pd.DataFrame({
        'timestamp': timestamps,
        'open': 100, 'high': 105, 'low': 95, 
        'close': 100.0,
        'volume': 1000
    })
    
    # Create a scenario where return is small (0.2%) < cost buffer (0.2% * 1.5 = 0.3%)
    # conf.RETURN_THRESHOLD usually 0.001. 
    # Cost buffer is 0.002. Threshold becomes 0.003
    
    # Set future return to 0.0025 (positive but below cost threshold)
    # Horizon is 3 (default).
    # We need to manually set TARGET_RET logic check
    # processor.process_for_training calls add_indicators then calcs target
    
    # Let's override conf temporarily if needed, but defaults might be fine.
    # If we set close prices to increase by 0.25% after 3 bars.
    
    horizon = getattr(conf, 'RETURN_HORIZON', 3)
    
    # We create a step change at index 50
    # From 50 to 50+horizon, price inc by 0.25%
    df.loc[50+horizon:, 'close'] = 100.25 
    
    # target at 50 should be 0.25% = 0.0025.
    # If cost aware (limit 0.003), this should be CLASS 1 (HOLD).
    # If not cost aware (limit 0.001), this would be CLASS 2 (BUY).
    
    # Need to pass feature_cols
    feature_cols = conf.FEATURE_COLS
    
    try:
        data_scaled, target_scaled, df_out, target_ret = processor.process_for_training(df, feature_cols)
        
        # Check index corresponding to original 50. 
        # Note: dropna happens. 
        # add_indicators drops (EMA 20, RSI 14). So first ~20 rows dropped.
        # target shift drops last `horizon`.
        
        # Let's align by timestamp or just inspect distribution
        # If max(target_ret) is 0.0025, and no class 2 exists, then success.
        
        max_ret = np.max(target_ret)
        print(f"Max return in data: {max_ret:.5f}")
        
        count_buy = np.sum(target_scaled == 2)
        print(f"Buy signals count: {count_buy}")
        
        if max_ret > 0.002 and count_buy == 0:
            print("✅ Cost-Aware Logic working (Small profit ignored).")
        elif count_buy > 0:
            print("❌ Cost-Aware Logic failed (Small profit triggered buy).")
        else:
            print("⚠️ Inconclusive (Maybe data didn't generate return).")
            
    except Exception as e:
        print(f"❌ Process failed: {e}")


def verify_fixes():
    print("\n[VERIFY] Bug Fixes Regression Test...")
    processor = DataProcessor()
    
    # 1. Test insufficient data for 1h features (should not crash)
    print("Testing insufficient data for 1H features...")
    df_small = pd.DataFrame({
        'timestamp': [pd.Timestamp.now()],
        'close': [100.0],
        'volume': [100.0]
    })
    try:
        df_res = processor.add_1h_features(df_small)
        if 'RSI_1H' in df_res.columns and pd.isna(df_res['RSI_1H'].iloc[0]):
             print("✅ Insufficient data handled gracefully (NaNs present).")
        else:
             print("⚠️ Columns missing or not NaN, but didn't crash.")
    except Exception as e:
        print(f"❌ Failed: {e}")

    # 2. Test check_data_gaps with Datetime objects (should not crash)
    print("Testing check_data_gaps with Datetime objects...")
    base = datetime(2023, 1, 1, 0, 0, 0)
    timestamps = [base.timestamp() * 1000 + i * 60000 * 15 for i in range(10)]
    df_dt = pd.DataFrame({'timestamp': pd.to_datetime(timestamps, unit='ms'), 'close': 100})
    try:
        gaps = check_data_gaps(df_dt, 15*60000)
        print(f"✅ check_data_gaps handled Datetime inputs. Gaps found: {len(gaps)}")
    except Exception as e:
        print(f"❌ check_data_gaps failed with Datetime: {e}")

if __name__ == "__main__":
    verify_data_integrity()
    verify_htf_features()
    verify_cost_aware_targets()
    verify_fixes()
