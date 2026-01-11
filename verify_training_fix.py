
import os
import shutil
import asyncio
import pandas as pd
import torch
from unittest.mock import MagicMock, patch
from src.bat.training import train_model
from src.bat.tui.app import App
from rich.console import Console

# Setup
TEST_DIR = "test_data_ts_fix"
DATA_PATH = os.path.join(TEST_DIR, "history.csv")
CHECKPOINT_PATH = "data/lstm_checkpoint.pth"

def setup_env():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    os.makedirs(TEST_DIR)
    # Mock config
    with patch("src.bat.data.dataset.conf") as mock_conf:
         mock_conf.FEATURE_COLS = ["close", "volume"] 
         mock_conf.SEQ_LENGTH = 10

def create_mock_data(rows=100, start_ts=1000000):
    data = {
        "timestamp": [start_ts + i * 60000 for i in range(rows)],
        "open": [100.0] * rows,
        "high": [105.0] * rows,
        "low": [95.0] * rows,
        "close": [100.0] * rows,
        "volume": [1000.0] * rows,
        "trans_count": [10] * rows
    }
    df = pd.DataFrame(data)
    df.to_csv(DATA_PATH, index=False)
    return df

def test_incremental_training():
    print("Testing Incremental Training & Timestamp Logic...")
    setup_env()
    
    # 1. Create initial data
    df1 = create_mock_data(rows=100, start_ts=1000000)
    max_ts_1 = int(df1["timestamp"].max())
    
    # 2. Train Phase 1 (2 epochs)
    print(f"Phase 1: Training on data ending at {max_ts_1}...")
    # Mock status/log callbacks
    mock_status = MagicMock()
    
    # We need to mock conf inside training.py or pass args. 
    # train_model uses 'conf' from imports. We'll rely on defaults or mock if needed.
    # To keep it simple, we assume the code works if we just run it, 
    # but we need to ensure paths are correct. 
    # The code writes to 'data/lstm_checkpoint.pth' hardcoded in training.py usually 
    # or uses a variable. Let's check training.py... It uses _checkpoint_path variable.
    
    with patch("src.bat.training.conf") as mock_conf:
        # Mock minimal config
        mock_conf.FEATURE_COLS = ["open", "high", "low", "close", "volume"]
        mock_conf.SEQ_LENGTH = 5
        mock_conf.LR = 0.001
        
        # Run training
        model, processor, _, last_ts = train_model(
            data_path=DATA_PATH,
            epochs=2,
            df=df1,
            on_status=mock_status
        )
        
        # Verify
        assert last_ts == max_ts_1, f"Expected last_ts {max_ts_1}, got {last_ts}"
        
        # Check checkpoint
        ckpt = torch.load(CHECKPOINT_PATH)
        assert ckpt["epoch"] == 2, f"Expected checkpoint epoch 2, got {ckpt['epoch']}"
        assert ckpt["meta"]["last_trained_timestamp"] == max_ts_1, "Checkpoint timestamp mismatch"
        print("Phase 1 Passed.")

    # 3. Add new data
    print("Phase 2: Adding new data...")
    df2 = create_mock_data(rows=200, start_ts=1000000) # 100 more rows
    max_ts_2 = int(df2["timestamp"].max())
    
    # 4. Train Phase 2 (2 more epochs)
    print(f"Phase 2: Training on data ending at {max_ts_2}...")
    
    with patch("src.bat.training.conf") as mock_conf:
        mock_conf.FEATURE_COLS = ["open", "high", "low", "close", "volume"]
        mock_conf.SEQ_LENGTH = 5
        mock_conf.LR = 0.001
        
        # This should load checkpoint (epoch 2) and train 2 MORE epochs -> 4
        model, processor, _, last_ts_2 = train_model(
            data_path=DATA_PATH,
            epochs=2,
            df=df2,
            on_status=mock_status
        )
        
        assert last_ts_2 == max_ts_2, f"Expected last_ts {max_ts_2}, got {last_ts_2}"
        
        ckpt = torch.load(CHECKPOINT_PATH)
        assert ckpt["epoch"] == 4, f"Expected checkpoint epoch 4 (2+2), got {ckpt['epoch']}"
        assert ckpt["meta"]["last_trained_timestamp"] == max_ts_2, "Checkpoint timestamp mismatch"
        print("Phase 2 Passed: Incremental training verified.")

def cleanup():
    if os.path.exists(TEST_DIR):
        shutil.rmtree(TEST_DIR)
    if os.path.exists("data/lstm_checkpoint.pth"):
        os.remove("data/lstm_checkpoint.pth")

if __name__ == "__main__":
    try:
        cleanup() # Pre-cleanup
        test_incremental_training()
        cleanup()
        print("\nALL TRAIN FIX TESTS PASSED!")
    except Exception as e:
        print(f"\nTEST FAILED: {e}")
        import traceback
        traceback.print_exc()
