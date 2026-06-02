from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardFold:
    train_indices: list[int]
    valid_indices: list[int]
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    valid_start: pd.Timestamp
    valid_end: pd.Timestamp


@dataclass(frozen=True)
class IndexWalkForwardFold:
    train_indices: list[int]
    valid_indices: list[int]


def parse_window(value: str | pd.Timedelta) -> pd.Timedelta:
    if isinstance(value, pd.Timedelta):
        return value
    return pd.Timedelta(value)


def walk_forward_splits(
    df: pd.DataFrame,
    train_window: str | pd.Timedelta,
    valid_window: str | pd.Timedelta,
    step_window: str | pd.Timedelta,
    purge_window: str | pd.Timedelta = "0d",
    embargo_window: str | pd.Timedelta = "0d",
    timestamp_col: str = "timestamp",
) -> list[WalkForwardFold]:
    if timestamp_col not in df.columns:
        raise ValueError(f"missing timestamp column: {timestamp_col}")

    ordered = df.sort_values(timestamp_col).reset_index(drop=False)
    timestamps = pd.to_datetime(ordered[timestamp_col], utc=True)
    train_delta = parse_window(train_window)
    valid_delta = parse_window(valid_window)
    step_delta = parse_window(step_window)
    purge_delta = parse_window(purge_window)
    embargo_delta = parse_window(embargo_window)
    if min(train_delta, valid_delta, step_delta) <= pd.Timedelta(0):
        raise ValueError("walk-forward windows must be positive")
    if min(purge_delta, embargo_delta) < pd.Timedelta(0):
        raise ValueError("walk-forward purge and embargo windows must be non-negative")

    folds: list[WalkForwardFold] = []
    cursor = timestamps.iloc[0]
    final_time = timestamps.iloc[-1]
    while cursor + train_delta < final_time:
        train_start = cursor
        train_end = train_start + train_delta
        valid_start = train_end + purge_delta + embargo_delta
        valid_end = valid_start + valid_delta

        train_mask = (timestamps >= train_start) & (timestamps < train_end)
        valid_mask = (timestamps >= valid_start) & (timestamps < valid_end)
        train_indices = ordered.loc[train_mask, "index"].astype(int).tolist()
        valid_indices = ordered.loc[valid_mask, "index"].astype(int).tolist()

        if train_indices and valid_indices:
            folds.append(
                WalkForwardFold(
                    train_indices=train_indices,
                    valid_indices=valid_indices,
                    train_start=train_start,
                    train_end=train_end,
                    valid_start=valid_start,
                    valid_end=valid_end,
                )
            )
        cursor = cursor + step_delta
    return folds


def purged_walk_forward_indices(
    length: int,
    train_size: int,
    valid_size: int,
    step_size: int,
    purge_size: int = 0,
    embargo_size: int = 0,
) -> list[IndexWalkForwardFold]:
    length = int(length)
    train_size = int(train_size)
    valid_size = int(valid_size)
    step_size = int(step_size)
    purge_size = max(int(purge_size), 0)
    embargo_size = max(int(embargo_size), 0)
    if min(length, train_size, valid_size, step_size) <= 0:
        raise ValueError("walk-forward sizes must be positive")

    folds: list[IndexWalkForwardFold] = []
    cursor = 0
    gap = purge_size + embargo_size
    while True:
        train_start = cursor
        train_end = train_start + train_size
        valid_start = train_end + gap
        valid_end = valid_start + valid_size
        if valid_end > length:
            break
        folds.append(
            IndexWalkForwardFold(
                train_indices=list(range(train_start, train_end)),
                valid_indices=list(range(valid_start, valid_end)),
            )
        )
        cursor += step_size
    return folds
