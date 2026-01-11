import pandas as pd
import asyncio
from datetime import datetime
from bat.logger import get_logger
from bat.config import conf
from bat.execution.spot_client import async_historical_klines

logger = get_logger("bat.data.integrity")

def check_data_gaps(df: pd.DataFrame, interval_ms: int) -> list[tuple[int, int]]:
    """
    Check for missing timestamps in the DataFrame.
    Returns a list of (start_gap_ms, end_gap_ms).
    """
    if df.empty:
        return []
    
    # Ensure sorted by timestamp
    df = df.sort_values("timestamp")
    
    # Convert to int64 (ms) for calculation
    # Only if it's datetime, convert to int (ns) -> ms
    if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        timestamps = df["timestamp"].astype('int64') // 10**6
    else:
        timestamps = df["timestamp"].astype(int).values
    
    gaps = []
    # Diff of timestamps should actally equal interval_ms
    diffs = timestamps[1:] - timestamps[:-1]
    
    # Identify indices where diff > interval_ms * 1.1 (tolerance)
    gap_indices = [i for i, d in enumerate(diffs) if d > interval_ms * 1.1] 
    
    for idx in gap_indices:
        # Gap starts after timestamps[idx] and ends before timestamps[idx+1]
        gap_start = timestamps[idx] + interval_ms
        gap_end = timestamps[idx+1] - interval_ms
        if gap_end >= gap_start:
            gaps.append((gap_start, gap_end))
            
    # Check for "Tail Gap" (Incremental Update)
    # If the last timestamp is older than Now - Interval, we need to fetch the latest data.
    if len(timestamps) > 0:
        last_ts = timestamps[-1]
        now_ts = int(datetime.now().timestamp() * 1000)
        
        # If gap is larger than 2 intervals, consider it missing
        if now_ts - last_ts > interval_ms * 2:
            # Start from last_ts + interval
            tail_gap_start = last_ts + interval_ms
            # End at now (fetch_klines typically handles the end time correctly)
            tail_gap_end = now_ts
            gaps.append((tail_gap_start, tail_gap_end))

    return gaps

async def heal_data_gaps(client, symbol: str, interval: str, gaps: list[tuple[int, int]]) -> list[dict]:
    """
    Fetch missing klines for the specified gaps.
    Returns a list of new klines chunks.
    """
    healed_klines = []
    for start_ms, end_ms in gaps:
        logger.info(f"Healing gap for {symbol} {interval}: {start_ms} to {end_ms}")
        
        start_str = datetime.fromtimestamp(start_ms/1000).strftime("%Y-%m-%d %H:%M:%S")
        end_str = datetime.fromtimestamp(end_ms/1000).strftime("%Y-%m-%d %H:%M:%S")
        
        try:
            klines = await async_historical_klines(client, symbol, interval, start_str, end_str)
            if klines:
                logger.info(f"Fetched {len(klines)} klines for gap.")
                healed_klines.extend(klines)
            else:
                logger.warning(f"No data found for gap {start_str} - {end_str}")
        except Exception as e:
            logger.error(f"Failed to heal gap {start_str}-{end_str}: {e}")
            
    return healed_klines

def merge_healed_data(original_df: pd.DataFrame, new_klines_list: list) -> pd.DataFrame:
    """
    Merge new klines into original dataframe and return sorted unique DF.
    """
    if not new_klines_list:
        return original_df

    new_df = pd.DataFrame(new_klines_list, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'q_vol', 'trades', 'tb_base', 'tb_quote', 'ignore'
    ])
    new_df['timestamp'] = pd.to_datetime(new_df['timestamp'], unit='ms')
    cols = ['open', 'high', 'low', 'close', 'volume']
    new_df[cols] = new_df[cols].astype(float)
    
    # Unify original_df timestamps
    if not pd.api.types.is_datetime64_any_dtype(original_df['timestamp']):
        try:
             # Assume ms if int/float, or parse strict
             original_df['timestamp'] = pd.to_datetime(original_df['timestamp'], unit='ms')
        except:
             pd.to_datetime(original_df['timestamp'])
    
    combined = pd.concat([original_df, new_df])
    # deduplicate by timestamp
    combined = combined.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
    
    return combined
