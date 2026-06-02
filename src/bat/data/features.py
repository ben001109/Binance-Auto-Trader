from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


OHLCV_COLUMNS = ["open", "high", "low", "close", "volume"]
MAX_CLOSE_JUMP_FRACTION = 5.0
MAX_HIGH_LOW_RATIO = 3.0
DEFAULT_RESEARCH_FEATURES = [
    "log_return_1",
    "log_return_3",
    "log_return_6",
    "log_return_12",
    "rolling_vol_12",
    "rolling_vol_48",
    "RSI_14",
    "MACD",
    "MACD_signal",
    "MACD_hist",
    "ATR_14",
    "BB_width",
    "volume_change",
    "volume_zscore",
]


@dataclass(frozen=True)
class DataQualityReport:
    rows_before: int
    rows_after: int
    duplicate_timestamps: int
    dropped_nan_rows: int
    missing_candles: int
    nan_count: int
    start_time: str | None
    end_time: str | None


def _normalize_timestamp(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_datetime(series, unit="ms", utc=True)
    return pd.to_datetime(series, utc=True)


def clean_ohlcv(
    df: pd.DataFrame,
    expected_interval: str | pd.Timedelta | None = None,
) -> tuple[pd.DataFrame, DataQualityReport]:
    required = ["timestamp", *OHLCV_COLUMNS]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"OHLCV data missing required columns: {', '.join(missing)}")

    rows_before = len(df)
    cleaned = df.copy()
    cleaned["timestamp"] = _normalize_timestamp(cleaned["timestamp"])
    for column in OHLCV_COLUMNS:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    if not np.isfinite(cleaned[OHLCV_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError("OHLCV values must be finite")

    cleaned = cleaned.sort_values("timestamp")
    duplicate_timestamps = int(cleaned.duplicated("timestamp").sum())
    cleaned = cleaned.drop_duplicates("timestamp", keep="last")

    before_dropna = len(cleaned)
    cleaned = cleaned.dropna(subset=required)
    dropped_nan_rows = before_dropna - len(cleaned)
    cleaned = cleaned.reset_index(drop=True)
    _validate_ohlcv_invariants(cleaned)

    missing_candles = 0
    if expected_interval is not None and len(cleaned) > 1:
        interval = pd.Timedelta(expected_interval)
        diffs = cleaned["timestamp"].diff().dropna()
        if interval > pd.Timedelta(0):
            missing_candles = int(
                sum(max(int(round(diff / interval)) - 1, 0) for diff in diffs)
            )

    report = DataQualityReport(
        rows_before=rows_before,
        rows_after=len(cleaned),
        duplicate_timestamps=duplicate_timestamps,
        dropped_nan_rows=dropped_nan_rows,
        missing_candles=missing_candles,
        nan_count=int(cleaned[required].isna().sum().sum()),
        start_time=cleaned["timestamp"].iloc[0].isoformat() if len(cleaned) else None,
        end_time=cleaned["timestamp"].iloc[-1].isoformat() if len(cleaned) else None,
    )
    return cleaned, report


def _validate_ohlcv_invariants(df: pd.DataFrame) -> None:
    if df.empty:
        return
    price_columns = ["open", "high", "low", "close"]
    if not np.isfinite(df[[*price_columns, "volume"]]).all().all():
        raise ValueError("OHLCV values must be finite")
    if (df[price_columns] <= 0).any().any():
        raise ValueError("OHLCV prices must be positive")
    if (df["volume"] < 0).any():
        raise ValueError("OHLCV volume must be non-negative")
    if (df["high"] < df["low"]).any():
        raise ValueError("OHLCV high must be greater than or equal to low")
    if ((df["open"] < df["low"]) | (df["open"] > df["high"])).any():
        raise ValueError("OHLCV open must be inside high/low range")
    if ((df["close"] < df["low"]) | (df["close"] > df["high"])).any():
        raise ValueError("OHLCV close must be inside high/low range")
    if ((df["high"] / df["low"]) > MAX_HIGH_LOW_RATIO).any():
        raise ValueError("OHLCV high/low outlier detected")
    close_jump = df["close"].pct_change().abs().dropna()
    if (close_jump > MAX_CLOSE_JUMP_FRACTION).any():
        raise ValueError("OHLCV close price outlier detected")


def _rsi(close: pd.Series, length: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(length).mean()
    loss = (-delta.clip(upper=0)).rolling(length).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.mask((loss == 0) & (gain > 0), 100.0)
    rsi = rsi.mask((gain == 0) & (loss > 0), 0.0)
    return rsi


def _atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    previous_close = df["close"].shift(1)
    ranges = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    )
    true_range = ranges.max(axis=1)
    return true_range.rolling(length).mean()


def add_research_features(df: pd.DataFrame) -> pd.DataFrame:
    featured = df.copy()
    for column in OHLCV_COLUMNS:
        featured[column] = pd.to_numeric(featured[column], errors="coerce")

    close = featured["close"]
    for period in (1, 3, 6, 12):
        featured[f"log_return_{period}"] = np.log(close / close.shift(period))

    featured["rolling_vol_12"] = featured["log_return_1"].rolling(12).std()
    featured["rolling_vol_48"] = featured["log_return_1"].rolling(48).std()
    featured["RSI_14"] = _rsi(close, 14)

    ema_fast = close.ewm(span=12, adjust=False).mean()
    ema_slow = close.ewm(span=26, adjust=False).mean()
    featured["MACD"] = ema_fast - ema_slow
    featured["MACD_signal"] = featured["MACD"].ewm(span=9, adjust=False).mean()
    featured["MACD_hist"] = featured["MACD"] - featured["MACD_signal"]
    featured["ATR_14"] = _atr(featured, 14)

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    featured["BB_width"] = ((bb_mid + 2 * bb_std) - (bb_mid - 2 * bb_std)) / bb_mid

    volume = featured["volume"]
    featured["volume_change"] = volume.pct_change()
    volume_mean = volume.rolling(20).mean()
    volume_std = volume.rolling(20).std()
    featured["volume_zscore"] = (volume - volume_mean) / (volume_std + 1e-8)
    return featured
