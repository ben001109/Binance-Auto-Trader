import math

import pandas as pd


def parse_timestamp_ms(value) -> int:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise ValueError("timestamp is empty")
    if isinstance(value, pd.Timestamp):
        ts = value.tz_convert("UTC") if value.tzinfo else value.tz_localize("UTC")
        return int(ts.timestamp() * 1000)
    if isinstance(value, (int, float)):
        return int(value)
    text = str(value).strip()
    if not text:
        raise ValueError("timestamp is empty")
    try:
        numeric = float(text)
    except ValueError:
        dt = pd.to_datetime(text, utc=True)
        return int(dt.timestamp() * 1000)
    return int(numeric)


def normalize_timestamp_series(values: pd.Series) -> pd.Series:
    timestamps = values.map(parse_timestamp_ms)
    return pd.to_datetime(timestamps, unit="ms", utc=True)
