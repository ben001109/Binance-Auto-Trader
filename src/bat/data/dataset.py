import pandas as pd
import pandas_ta as ta
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
import numpy as np

from bat.config import conf

class DataProcessor:
    def __init__(self):
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def add_indicators(self, df):
        df = df.copy()
        # 確保數據是數值型
        cols = ['open', 'high', 'low', 'close', 'volume']
        df[cols] = df[cols].astype(float)

        # 加入指標
        rsi = df.ta.rsi(length=14)
        if isinstance(rsi, pd.DataFrame):
            rsi = rsi.iloc[:, 0]
        df['RSI'] = rsi
        ema = df.ta.ema(length=20)
        if isinstance(ema, pd.DataFrame):
            ema = ema.iloc[:, 0]
        df['EMA_20'] = ema
        df['RET_1'] = df['close'].pct_change()
        df['VOL_20'] = df['RET_1'].rolling(20).std()
        vol_mean = df['volume'].rolling(20).mean()
        vol_std = df['volume'].rolling(20).std()
        df['VOL_Z'] = (df['volume'] - vol_mean) / vol_std.replace(0, np.nan)
        df['VOL_Z'] = df['VOL_Z'].fillna(0.0)
        
        # Add Higher Timeframe (1h) Features
        df = self.add_1h_features(df)
        
        df.dropna(inplace=True)
        return df

    def add_1h_features(self, df):
        # Only meaningful if index is DatetimeIndex or 'timestamp' column exists
        # Our df usually has 'timestamp' column as datetime (from BaseStrategy) or process methods
        # In process_for_inference/training, we see BaseStrategy converts to datetime.
        # But here add_indicators expects ready df. Let's check timestamp.
        
        if 'timestamp' not in df.columns:
            # If loaded from csv, it might be string or not parsed yet? 
            # In BaseStrategy.fetch_data we convert to datetime.
            return df

        # Initialize default columns (NaN) to prevent KeyError in early exits
        df['RSI_1H'] = np.nan
        df['EMA_20_1H'] = np.nan
            
        # Create temp series with datetime index
        # Create temp series with datetime index or ensure type
        # Check current type
        is_dt = pd.api.types.is_datetime64_any_dtype(df['timestamp'])
        if not is_dt:
             try:
                 # Try numeric ms first (most common in this app)
                 # fix(logic): explicit UTC to match integrity check
                 df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
             except Exception:
                 try:
                     # Fallback to auto-parse (for strings)
                     df['timestamp'] = pd.to_datetime(df['timestamp'])
                 except Exception:
                     # Failed to convert, cannot resample
                     return df

        # Resample logic
        # We set index to timestamp temporarily
        df_temp = df.set_index('timestamp').sort_index()
        
        if not isinstance(df_temp.index, pd.DatetimeIndex):
            return df
        
        # Resample to 1h
        try:
            df_1h = df_temp['close'].resample('1h').last()
        except TypeError:
            # Final safety net
            return df
        
        # Calc 1h indicators
        rsi_1h = ta.rsi(df_1h, length=14)
        ema_1h = ta.ema(df_1h, length=20)
        
        # Check if indicators were calculated successfully
        # pandas-ta can return None or empty Series if data is insufficient
        is_invalid = False
        if rsi_1h is None or ema_1h is None:
            is_invalid = True
        elif isinstance(rsi_1h, (pd.Series, pd.DataFrame)) and rsi_1h.empty:
            is_invalid = True
        elif isinstance(ema_1h, (pd.Series, pd.DataFrame)) and ema_1h.empty:
            is_invalid = True
            
        if is_invalid:
             # Create NaN columns matching original index
             df['RSI_1H'] = np.nan
             df['EMA_20_1H'] = np.nan
             return df

        # Shift 1 to avoid lookahead (use closed candle data)
        rsi_1h = rsi_1h.shift(1)
        ema_1h = ema_1h.shift(1)
        
        # Map back to original timeframe (15m is inside the NEXT hour? No.)
        # If we have 15m candles: 10:00, 10:15, 10:30, 10:45.
        # At 10:00, the 9:00-10:00 candle just closed. So rsi_1h (9-10) is available.
        # The resample('1h').last() puts label at 9:00? or 10:00?
        # Default label is left (start of bin). 9:00 bin contains 9:00-10:00 data.
        # If label='left', 9:00 row has close of 9:59.
        # So at 10:00 (timestamp), we can use data from 9:00 label.
        
        # Reindex/Forward fill
        # This reindexes 1h series to match 15m index, ffilling values
        rsi_reindexed = rsi_1h.reindex(df_temp.index, method='ffill')
        ema_reindexed = ema_1h.reindex(df_temp.index, method='ffill')
        
        df['RSI_1H'] = rsi_reindexed.values
        df['EMA_20_1H'] = ema_reindexed.values
        
        return df

    def process_for_training(self, df, feature_cols):
        df = self.add_indicators(df)
        horizon = max(1, int(conf.RETURN_HORIZON))
        df['TARGET_RET'] = df['close'].pct_change(periods=horizon).shift(-horizon)
        df.dropna(inplace=True)
        data = df[feature_cols].values
        target_ret = df['TARGET_RET'].values
        
        # Cost-Aware Threshold logic
        # Fee is ~0.1% (maker/taker) * 2 (entry/exit) = 0.002
        cost_buffer = 0.002
        threshold = max(conf.RETURN_THRESHOLD, cost_buffer * 1.5) # ensure threshold covers cost
        target_class = np.where(
            target_ret > threshold,
            2,
            np.where(target_ret < -threshold, 0, 1),
        )

        # fix(error): prevent fit on empty data
        if len(data) == 0:
            # Return empty structure or raise specific error caught by analyzer
            raise ValueError("Insufficient data for training after processing (0 samples)")

        self.scaler.fit(data)
        data_scaled = self.scaler.transform(data)
        return data_scaled, target_class.astype(np.int64), df, target_ret

    def process_for_inference(self, df, feature_cols):
        df = self.add_indicators(df)
        data = df[feature_cols].values
        data_scaled = self.scaler.transform(data)
        return data_scaled, df

class TimeSeriesDataset(Dataset):
    def __init__(self, data, targets, seq_length, weights=None):
        self.data = data
        self.targets = targets
        self.seq_length = seq_length
        self.weights = weights

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, index):
        # X: 過去 seq_length 筆數據
        x = self.data[index : index + self.seq_length]
        y = self.targets[index + self.seq_length - 1]
        if self.weights is None:
            w = 1.0
        else:
            w = float(self.weights[index + self.seq_length - 1])
        return torch.FloatTensor(x), torch.LongTensor([y]), torch.FloatTensor([w])


def build_cost_aware_labels(
    df: pd.DataFrame,
    horizon: int,
    fee: float,
    slippage: float,
    min_edge: float,
    close_col: str = "close",
) -> pd.Series:
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    future_return = df[close_col].shift(-horizon) / df[close_col] - 1
    threshold = float(fee) + float(slippage) + float(min_edge)
    labels = pd.Series(pd.NA, index=df.index, dtype="Int64")
    known = future_return.notna()
    labels.loc[known] = 1
    labels.loc[future_return > threshold] = 2
    labels.loc[future_return < -threshold] = 0
    return labels


class ResearchSequenceDataset(Dataset):
    def __init__(self, features, labels, seq_len: int, weights=None):
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        self.features = np.asarray(features, dtype=np.float32)
        self.labels = np.asarray(labels)
        self.seq_len = int(seq_len)
        self.weights = None if weights is None else np.asarray(weights, dtype=np.float32)
        if len(self.features) != len(self.labels):
            raise ValueError("features and labels must have the same length")
        if self.weights is not None and len(self.weights) != len(self.labels):
            raise ValueError("weights and labels must have the same length")

    def __len__(self):
        return max(len(self.features) - self.seq_len + 1, 0)

    def __getitem__(self, index):
        end = index + self.seq_len
        label_index = end - 1
        x = torch.from_numpy(self.features[index:end].astype(np.float32, copy=False))
        y = torch.tensor(int(self.labels[label_index]), dtype=torch.long)
        if self.weights is None:
            return x, y
        w = torch.tensor(float(self.weights[label_index]), dtype=torch.float32)
        return x, y, w
