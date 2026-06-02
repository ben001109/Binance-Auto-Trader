from __future__ import annotations

import pandas as pd


def rolling_zscore_normalize(
    df: pd.DataFrame,
    feature_cols: list[str],
    window: int,
    epsilon: float = 1e-8,
) -> pd.DataFrame:
    if window <= 1:
        raise ValueError("rolling normalization window must be greater than 1")
    normalized = df.copy()
    rolling = normalized[feature_cols].rolling(window)
    mean = rolling.mean().shift(1)
    std = rolling.std().shift(1)
    normalized[feature_cols] = (normalized[feature_cols] - mean) / (std + epsilon)
    return normalized
