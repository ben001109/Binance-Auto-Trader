import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import subprocess
import shutil
import os
from pathlib import Path

from dataclasses import dataclass

from bat.config import conf
from bat.models.lstm import CryptoLSTM
from bat.models.factory import create_model, normalize_model_name
from bat.data.dataset import DataProcessor, TimeSeriesDataset
from bat.data.split import IndexWalkForwardFold, purged_walk_forward_indices
from bat.data.timestamps import parse_timestamp_ms
from bat.backtest import run_backtest
from bat.research_config import ResearchConfig, load_research_config, with_model_name
from bat.services.dataset_service import DatasetService
from bat.services.model_admission import assess_model_admission
from bat.services.run_manager import RunManager, TrainingRun
from bat.services.artifact_security import (
    ArtifactSecurityError,
    load_manifested_torch_state,
    safe_torch_load,
    write_artifact_manifest,
)
from bat.training_metrics import classification_report, trading_report
import numpy as np
from bat.logger import get_logger


SKLEARN_MODEL_NAMES = {"histgb"}


@dataclass
class RiskParams:
    stop_loss: float
    take_profit: float
    max_dd_stop: float
    position_splits: int


@dataclass
class TrainingInputs:
    data_scaled: np.ndarray
    target_scaled: np.ndarray
    df: pd.DataFrame
    target_ret: np.ndarray
    processor: DataProcessor | None
    feature_cols: list[str]
    seq_length: int
    last_trained_timestamp: int
    pipeline: str


@dataclass
class EarlyStoppingState:
    metric_name: str
    patience: int
    best_metric: float | None = None
    best_epoch: int = 0
    bad_epochs: int = 0
    improved: bool = False
    should_stop: bool = False
    stop_reason: str | None = None


_stop_training = False
_checkpoint_path = "data/lstm_checkpoint.pth"


def set_stop_training(value: bool) -> None:
    global _stop_training
    _stop_training = value


def should_stop_training() -> bool:
    return _stop_training


def _load_training_data(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)
    required_cols = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Training data missing required columns: {missing_str}")
    for col in required_cols:
        df[col] = df[col].astype(float)
    return df


def _normalize_data_pipeline(value: str | None) -> str:
    normalized = str(value or "legacy").strip().lower().replace("_", "-")
    if normalized in {"research", "research-pipeline"}:
        return "research"
    return "legacy"


def _latest_timestamp_from_df(df: pd.DataFrame) -> int:
    def _latest_from_series(series: pd.Series) -> int:
        values = []
        for value in series.dropna():
            try:
                values.append(parse_timestamp_ms(value))
            except Exception:
                continue
        return max(values) if values else 0

    if "timestamp" in df.columns:
        latest = _latest_from_series(df["timestamp"])
        if latest:
            return latest
    if "close_time" in df.columns:
        return _latest_from_series(df["close_time"])
    return 0


def _timestamp_ms_from_iso(value: str | None) -> int:
    if not value:
        return 0
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return 0
    return int(parsed.value // 1_000_000)


def _prepare_training_inputs(
    data_path: str,
    df: pd.DataFrame | None = None,
    data_pipeline: str | None = None,
    research_config: ResearchConfig | None = None,
) -> TrainingInputs:
    config = research_config or load_research_config()
    pipeline = _normalize_data_pipeline(
        data_pipeline if data_pipeline is not None else config.training.data_pipeline
    )
    if pipeline == "research":
        service = DatasetService(config)
        if df is None and config.training.use_preprocessing_cache:
            prepared = service.prepare_csv_cached(data_path)
            source_df = pd.DataFrame()
            last_trained_timestamp = _timestamp_ms_from_iso(prepared.quality_report.end_time)
        else:
            source_df = df if df is not None else _load_training_data(data_path)
            last_trained_timestamp = _latest_timestamp_from_df(source_df)
            prepared = service.prepare_dataframe(source_df)
        if len(prepared.features) == 0:
            raise ValueError("Insufficient data after research pipeline preprocessing")
        return TrainingInputs(
            data_scaled=prepared.features,
            target_scaled=prepared.labels.astype(np.int64),
            df=source_df,
            target_ret=prepared.future_returns,
            processor=None,
            feature_cols=list(prepared.feature_columns),
            seq_length=config.data.seq_len,
            last_trained_timestamp=last_trained_timestamp,
            pipeline="research",
        )

    source_df = df if df is not None else _load_training_data(data_path)
    last_trained_timestamp = _latest_timestamp_from_df(source_df)
    processor = DataProcessor()
    data_scaled, target_scaled, processed_df, target_ret = processor.process_for_training(
        source_df,
        conf.FEATURE_COLS,
    )
    return TrainingInputs(
        data_scaled=data_scaled,
        target_scaled=target_scaled,
        df=processed_df,
        target_ret=target_ret,
        processor=processor,
        feature_cols=list(conf.FEATURE_COLS),
        seq_length=conf.SEQ_LENGTH,
        last_trained_timestamp=last_trained_timestamp,
        pipeline="legacy",
    )


def _resolve_training_model_config(
    config: ResearchConfig,
    model_name: str | None = None,
) -> ResearchConfig:
    if model_name is None:
        normalize_model_name(config.model.name)
        return config
    return with_model_name(config, normalize_model_name(model_name))


def _label_distribution(targets: np.ndarray) -> dict[int, int]:
    flat = np.asarray(targets).astype(int).reshape(-1)
    return {label: int((flat == label).sum()) for label in (0, 1, 2)}


def _compute_training_weights(
    targets: np.ndarray,
    target_ret: np.ndarray,
    class_weight_strength: float = 1.0,
) -> np.ndarray:
    labels = np.asarray(targets).astype(int).reshape(-1)
    returns = np.asarray(target_ret, dtype=float).reshape(-1)
    if len(labels) == 0:
        return np.asarray([], dtype=float)
    if len(returns) != len(labels):
        raise ValueError("target_ret must have the same length as targets")

    class_counts = np.bincount(labels, minlength=3).astype(float)
    class_counts[class_counts == 0] = 1.0
    inverse_frequency = class_counts.sum() / (3.0 * class_counts)
    strength = min(max(float(class_weight_strength), 0.0), 1.0)
    class_weights = 1.0 + strength * (inverse_frequency - 1.0)

    abs_ret = np.abs(returns)
    scale = np.percentile(abs_ret, 90) if len(abs_ret) else 0.0
    if scale <= 0:
        sample_weights = np.ones_like(abs_ret, dtype=float)
    else:
        sample_weights = 1.0 + np.clip(abs_ret / scale, 0.0, 3.0)
    return sample_weights * class_weights[labels]


def _create_training_run(
    config: ResearchConfig,
    model_name: str,
    feature_cols: list[str],
    targets: np.ndarray,
    runs_root: str | os.PathLike = "runs",
    timestamp: str | None = None,
) -> TrainingRun:
    return RunManager(root=runs_root).create_run(
        config=config,
        model_name=model_name,
        feature_columns=list(feature_cols),
        label_distribution=_label_distribution(targets),
        timestamp=timestamp,
    )


def _save_run_best_model(run: TrainingRun | None, model) -> None:
    if run is not None:
        run.save_best_model(model)


def _save_run_best_sklearn_model(run: TrainingRun | None, model) -> None:
    if run is not None:
        run.save_sklearn_model(model)


def _run_artifact_path(run, filename: str) -> Path:
    direct_name = "best_model_path" if filename == "best_model.pt" else "best_sklearn_model_path"
    direct = getattr(run, direct_name, None)
    if direct is not None:
        return Path(direct)
    return Path(getattr(run, "path")) / filename


def _update_early_stopping(
    state: EarlyStoppingState | None,
    metrics: dict,
    *,
    epoch: int,
    metric_name: str,
    patience: int,
) -> EarlyStoppingState:
    state = state or EarlyStoppingState(metric_name=metric_name, patience=patience)
    if metric_name not in metrics or metrics.get(metric_name) is None:
        return EarlyStoppingState(
            metric_name=metric_name,
            patience=patience,
            best_metric=state.best_metric,
            best_epoch=state.best_epoch,
            bad_epochs=state.bad_epochs,
            improved=False,
            should_stop=False,
            stop_reason=f"metric {metric_name} unavailable",
        )

    value = float(metrics[metric_name])
    if value != value:
        return EarlyStoppingState(
            metric_name=metric_name,
            patience=patience,
            best_metric=state.best_metric,
            best_epoch=state.best_epoch,
            bad_epochs=state.bad_epochs,
            improved=False,
            should_stop=False,
            stop_reason=f"metric {metric_name} unavailable",
        )

    lower_is_better = metric_name.endswith("loss")
    improved = state.best_metric is None or (
        value < state.best_metric if lower_is_better else value > state.best_metric
    )
    if improved:
        return EarlyStoppingState(
            metric_name=metric_name,
            patience=patience,
            best_metric=value,
            best_epoch=epoch,
            bad_epochs=0,
            improved=True,
            should_stop=False,
            stop_reason=None,
        )

    bad_epochs = state.bad_epochs + 1
    should_stop = bad_epochs >= max(int(patience), 0)
    return EarlyStoppingState(
        metric_name=metric_name,
        patience=patience,
        best_metric=state.best_metric,
        best_epoch=state.best_epoch,
        bad_epochs=bad_epochs,
        improved=False,
        should_stop=should_stop,
        stop_reason=(
            f"{metric_name} did not improve for {bad_epochs} epochs"
            if should_stop
            else None
        ),
    )


def _time_ordered_train_valid_indices(length: int, valid_ratio: float = 0.2) -> tuple[list[int], list[int]]:
    if length <= 0:
        return [], []
    ratio = min(max(float(valid_ratio), 0.0), 0.9)
    valid_size = int(round(length * ratio))
    if valid_size <= 0 and length > 1:
        valid_size = 1
    if valid_size >= length:
        valid_size = max(length - 1, 0)
    split = length - valid_size
    return list(range(split)), list(range(split, length))


def _validation_folds(length: int, config: ResearchConfig) -> list[IndexWalkForwardFold]:
    method = str(config.validation.method or "holdout").strip().lower().replace("-", "_")
    if method != "walk_forward":
        train_indices, valid_indices = _time_ordered_train_valid_indices(
            length,
            valid_ratio=config.validation.holdout_ratio,
        )
        return [IndexWalkForwardFold(train_indices, valid_indices)] if train_indices and valid_indices else []

    valid_size = max(int(round(length * float(config.validation.holdout_ratio))), 1)
    train_size = max(valid_size * 2, int(round(length * (1.0 - float(config.validation.holdout_ratio)) * 0.5)))
    train_size = min(train_size, max(length - valid_size - int(config.data.horizon), 1))
    folds = purged_walk_forward_indices(
        length=length,
        train_size=train_size,
        valid_size=valid_size,
        step_size=valid_size,
        purge_size=max(int(config.data.horizon), 0),
        embargo_size=max(int(config.data.horizon), 0),
    )
    if folds:
        return folds
    train_indices, valid_indices = _time_ordered_train_valid_indices(
        length,
        valid_ratio=config.validation.holdout_ratio,
    )
    if train_indices and valid_indices:
        gap = max(int(config.data.horizon), 0) * 2
        valid_start = min(valid_indices)
        train_indices = [index for index in train_indices if index + gap < valid_start]
    return [IndexWalkForwardFold(train_indices, valid_indices)] if train_indices and valid_indices else []


def _non_overlapping_validation_folds(
    train_indices: list[int],
    validation_folds: list[IndexWalkForwardFold],
) -> list[IndexWalkForwardFold]:
    train_set = set(train_indices)
    return [fold for fold in validation_folds if train_set.isdisjoint(fold.valid_indices)]


def _aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    if not fold_metrics:
        return {}
    aggregate: dict = {"fold_count": len(fold_metrics)}
    distributions = {"label_distribution", "prediction_distribution"}
    summed = {"trade_count", "filtered_to_hold"}
    worst_min = {"max_drawdown"}

    keys = set().union(*(metrics.keys() for metrics in fold_metrics))
    for key in sorted(keys):
        if key == "fold":
            continue
        values = [metrics[key] for metrics in fold_metrics if key in metrics]
        if key in distributions:
            combined = {0: 0, 1: 0, 2: 0}
            for distribution in values:
                if isinstance(distribution, dict):
                    for label, count in distribution.items():
                        combined[int(label)] = combined.get(int(label), 0) + int(count)
            aggregate[key] = {label: combined.get(label, 0) for label in (0, 1, 2)}
        elif key in summed:
            aggregate[key] = int(sum(int(value or 0) for value in values))
        elif key in worst_min:
            aggregate[key] = float(min(float(value or 0.0) for value in values))
        elif all(isinstance(value, (int, float, np.integer, np.floating)) for value in values):
            aggregate[key] = float(np.mean([float(value) for value in values]))
    return aggregate


def _aligned_future_returns(target_ret: np.ndarray, seq_length: int, dataset_length: int) -> np.ndarray:
    values = np.asarray(target_ret, dtype=float).reshape(-1)
    start = max(int(seq_length) - 1, 0)
    aligned = values[start : start + dataset_length]
    if len(aligned) < dataset_length:
        padded = np.zeros(dataset_length, dtype=float)
        padded[: len(aligned)] = aligned
        return padded
    return aligned


def _apply_decision_filter(
    logits,
    confidence_threshold: float = 0.0,
    edge_threshold: float = 0.0,
) -> tuple[np.ndarray, int]:
    probs = torch.softmax(logits.detach().cpu(), dim=1)
    confidence, predictions = probs.max(dim=1)
    confidence_threshold = max(float(confidence_threshold), 0.0)
    edge_threshold = max(float(edge_threshold), 0.0)
    action_predictions = predictions != 1
    low_confidence_actions = action_predictions & (confidence < confidence_threshold)
    sell_edge = probs[:, 0] - probs[:, 2]
    buy_edge = probs[:, 2] - probs[:, 0]
    weak_edge_actions = ((predictions == 0) & (sell_edge < edge_threshold)) | (
        (predictions == 2) & (buy_edge < edge_threshold)
    )
    filtered_actions = low_confidence_actions | weak_edge_actions
    predictions = predictions.clone()
    predictions[filtered_actions] = 1
    filtered_count = int(filtered_actions.sum().item())
    return predictions.numpy().astype(int), filtered_count


def _sklearn_probabilities_3(model, features: np.ndarray) -> np.ndarray:
    raw_probabilities = model.predict_proba(features)
    probabilities = np.zeros((len(features), 3), dtype=float)
    for column, label in enumerate(model.classes_):
        label_int = int(label)
        if 0 <= label_int < 3:
            probabilities[:, label_int] = raw_probabilities[:, column]
    return probabilities


def _apply_decision_filter_to_probabilities(
    probabilities: np.ndarray,
    confidence_threshold: float = 0.0,
    edge_threshold: float = 0.0,
) -> tuple[np.ndarray, int]:
    logits = torch.log(torch.as_tensor(np.clip(probabilities, 1e-12, 1.0), dtype=torch.float32))
    return _apply_decision_filter(
        logits,
        confidence_threshold=confidence_threshold,
        edge_threshold=edge_threshold,
    )


def _train_sklearn_histgb(
    *,
    inputs: TrainingInputs,
    weights: np.ndarray,
    model_config: ResearchConfig,
    model_name_for_meta: str,
    epochs: int,
    on_log=None,
    on_status=None,
):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import log_loss

    dataset_length = max(len(inputs.data_scaled) - int(inputs.seq_length), 0)
    if dataset_length < 2:
        raise ValueError("Insufficient data for histgb training after sequence alignment")

    start = max(int(inputs.seq_length) - 1, 0)
    end = start + dataset_length
    features = np.asarray(inputs.data_scaled[start:end], dtype=np.float32)
    labels = np.asarray(inputs.target_scaled[start:end], dtype=np.int64)
    sample_weights = np.asarray(weights[start:end], dtype=float)
    future_returns = _aligned_future_returns(inputs.target_ret, inputs.seq_length, dataset_length)
    folds = _validation_folds(dataset_length, model_config)
    if not folds:
        raise ValueError("histgb training requires non-empty train and validation splits")

    training_run = _create_training_run(
        config=model_config,
        model_name=model_name_for_meta,
        feature_cols=inputs.feature_cols,
        targets=inputs.target_scaled,
    )
    if on_log:
        on_log(f">>> [訓練] sklearn baseline model={model_name_for_meta}")
        on_log(f">>> [訓練] Run directory: {training_run.path}")

    fold_metrics = []
    model = None
    for fold_index, fold in enumerate(folds, start=1):
        model = HistGradientBoostingClassifier(
            max_iter=max(int(epochs), 1),
            learning_rate=float(model_config.training.lr),
            random_state=42,
        )
        x_train = features[fold.train_indices]
        y_train = labels[fold.train_indices]
        model.fit(x_train, y_train, sample_weight=sample_weights[fold.train_indices])

        train_probabilities = np.clip(_sklearn_probabilities_3(model, x_train), 1e-12, 1.0)
        x_valid = features[fold.valid_indices]
        y_valid = labels[fold.valid_indices]
        valid_probabilities = np.clip(_sklearn_probabilities_3(model, x_valid), 1e-12, 1.0)
        predictions, filtered_to_hold = _apply_decision_filter_to_probabilities(
            valid_probabilities,
            confidence_threshold=model_config.validation.decision_confidence_threshold,
            edge_threshold=model_config.validation.decision_edge_threshold,
        )
        class_metrics = classification_report(y_valid, predictions)
        returns_for_valid = future_returns[fold.valid_indices]
        trade_metrics = trading_report(
            predictions,
            returns_for_valid,
            fee=model_config.label.fee,
            slippage=model_config.label.slippage,
        )
        fold_metrics.append(
            {
                "fold": fold_index,
                "train_loss": float(log_loss(y_train, train_probabilities, labels=[0, 1, 2])),
                "val_loss": float(log_loss(y_valid, valid_probabilities, labels=[0, 1, 2])),
                "val_accuracy": class_metrics["accuracy"],
                "val_macro_f1": class_metrics["macro_f1"],
                "label_distribution": class_metrics["label_distribution"],
                "prediction_distribution": class_metrics["prediction_distribution"],
                "filtered_to_hold": filtered_to_hold,
                **trade_metrics,
            }
        )

    metrics = {
        "epoch": 1,
        **_aggregate_fold_metrics(fold_metrics),
        "validation_method": str(model_config.validation.method),
        "decision_confidence_threshold": float(model_config.validation.decision_confidence_threshold),
        "decision_edge_threshold": float(model_config.validation.decision_edge_threshold),
    }
    training_run.write_epoch_metrics(metrics)
    _save_run_best_sklearn_model(training_run, model)
    admission = assess_model_admission(metrics, [_run_artifact_path(training_run, "best_model.pkl")])
    best_metric = metrics.get(model_config.training.early_stopping_metric)
    training_run.write_report(
        {
            "status": "completed",
            "model_family": "sklearn",
            "monitored_metric": model_config.training.early_stopping_metric,
            "patience": model_config.training.early_stopping_patience,
            "completed_epochs": 1,
            "best_epoch": 1,
            "best_metric": best_metric,
            "stop_reason": None,
            "validation_method": str(model_config.validation.method),
            "fold_count": metrics.get("fold_count", 1),
            "fold_metrics": fold_metrics,
            "admission": admission.to_dict(),
        }
    )
    if on_status:
        on_status({"epoch": 1, "epochs": 1, "progress": 1.0, **metrics})
    if on_log:
        on_log(
            f">>> [訓練] HistGB val_macro_f1={metrics['val_macro_f1']:.4f} "
            f"expected_ret={metrics['expected_return_after_cost']:.6f}"
        )
    return model, inputs.processor, inputs.df, inputs.last_trained_timestamp


def evaluate_model_on_dataset(
    model,
    dataset,
    indices: list[int],
    device: torch.device,
    fee: float,
    slippage: float,
    future_returns=None,
    batch_size: int = 256,
    confidence_threshold: float = 0.0,
    edge_threshold: float = 0.0,
) -> dict:
    if not indices:
        return {}
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    was_training = model.training
    model.eval()
    losses = []
    labels = []
    predictions = []
    filtered_to_hold = 0
    with torch.no_grad():
        for x_batch, y_batch, _w_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device).squeeze(1)
            logits = model(x_batch)
            losses.append(float(criterion(logits, y_batch).item()))
            labels.extend(y_batch.detach().cpu().numpy().astype(int).tolist())
            batch_predictions, batch_filtered = _apply_decision_filter(
                logits,
                confidence_threshold=confidence_threshold,
                edge_threshold=edge_threshold,
            )
            predictions.extend(batch_predictions.tolist())
            filtered_to_hold += batch_filtered
    if was_training:
        model.train()

    class_metrics = classification_report(np.asarray(labels), np.asarray(predictions))
    if future_returns is None:
        returns_for_indices = np.zeros(len(predictions), dtype=float)
    else:
        returns = np.asarray(future_returns, dtype=float).reshape(-1)
        returns_for_indices = returns[indices] if len(returns) >= max(indices) + 1 else np.zeros(len(predictions))
    trade_metrics = trading_report(
        np.asarray(predictions),
        returns_for_indices,
        fee=fee,
        slippage=slippage,
    )
    return {
        "val_loss": float(np.mean(losses)) if losses else 0.0,
        "val_accuracy": class_metrics["accuracy"],
        "val_macro_f1": class_metrics["macro_f1"],
        "label_distribution": class_metrics["label_distribution"],
        "prediction_distribution": class_metrics["prediction_distribution"],
        "filtered_to_hold": filtered_to_hold,
        "decision_confidence_threshold": float(confidence_threshold),
        "decision_edge_threshold": float(edge_threshold),
        **trade_metrics,
    }


def _create_sequences(data: np.ndarray, seq_length: int):
    xs = []
    for i in range(len(data) - seq_length):
        xs.append(data[i : i + seq_length])
    return np.array(xs)


def _log_baseline_stats(logger, targets: np.ndarray) -> None:
    if targets is None or len(targets) == 0:
        return
    flat = np.asarray(targets).reshape(-1)
    counts = np.bincount(flat, minlength=3)
    total = counts.sum()
    if total == 0:
        return
    majority = counts.argmax()
    majority_acc = counts[majority] / total
    probs = counts / total
    random_acc = float((probs ** 2).sum())
    logger.info(
        "Baseline: class_counts=%s majority_class=%s majority_acc=%.4f random_acc=%.4f",
        counts.tolist(),
        int(majority),
        majority_acc,
        random_acc,
    )


def _log_return_alignment(
    logger,
    model: CryptoLSTM,
    processor: DataProcessor,
    data_scaled: np.ndarray,
    df: pd.DataFrame,
    max_samples: int = 5000,
) -> None:
    if len(data_scaled) <= conf.SEQ_LENGTH:
        return
    total_sequences = len(data_scaled) - conf.SEQ_LENGTH
    if total_sequences <= 0:
        return

    # Sample indices FIRST to avoid OOM
    if total_sequences > max_samples:
        indices = np.linspace(0, total_sequences - 1, max_samples).astype(int)
    else:
        indices = np.arange(total_sequences)

    sequences = []
    actual_returns = []
    
    # Only create sequences for sampled indices
    target_values = df["TARGET_RET"].values
    for idx in indices:
        sequences.append(data_scaled[idx : idx + conf.SEQ_LENGTH])
        # target return is at idx + seq_length - 1 (since sequence ends at idx+seq_length)
        # Actually in original code: df["TARGET_RET"].values[conf.SEQ_LENGTH - 1 :]
        # So alignment: sequence starting at i (ending at i+seq_len) corresponds to target at i+seq_len-1
        actual_returns.append(target_values[idx + conf.SEQ_LENGTH - 1])

    sequences = np.array(sequences)
    actual_returns = np.array(actual_returns)
    preds = []
    batch_size = 256
    model.eval()
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        x = torch.from_numpy(batch).float().to(conf.DEVICE)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds.append(probs)
    probs = np.vstack(preds) if preds else np.zeros((0, 3))
    if probs.size == 0:
        return
    expected_returns = (probs[:, 2] - probs[:, 0]) * conf.RETURN_THRESHOLD
    mean_expected = float(np.mean(expected_returns))
    mean_actual = float(np.mean(actual_returns))
    corr = float(np.corrcoef(expected_returns, actual_returns)[0, 1]) if len(expected_returns) > 1 else 0.0
    hit_rate = float(np.mean(np.sign(expected_returns) == np.sign(actual_returns)))
    logger.info(
        "Return alignment: mean_expected=%.6f mean_actual=%.6f corr=%.4f hit_rate=%.4f",
        mean_expected,
        mean_actual,
        corr,
        hit_rate,
    )


def _log_gpu_stats(logger, epoch, batch_idx):
    device_backend = getattr(conf, "DEVICE_BACKEND", None)
    if not isinstance(device_backend, str):
        device_backend = "rocm" if getattr(getattr(torch, "version", None), "hip", None) else "cuda"
    if device_backend not in {"cuda", "rocm"} or not torch.cuda.is_available():
        return
    device_name = getattr(conf, "DEVICE_NAME", None)
    if not isinstance(device_name, str):
        device_name = "AMD ROCm" if device_backend == "rocm" else "NVIDIA CUDA"
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    logger.debug(
        "%s memory: allocated=%.2fMB reserved=%.2fMB (epoch=%s batch=%s)",
        device_name,
        allocated,
        reserved,
        epoch,
        batch_idx,
    )
    smi_command = None
    smi_label = None
    if device_backend == "cuda" and shutil.which("nvidia-smi"):
        smi_command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ]
        smi_label = "NVIDIA GPU"
    elif device_backend == "rocm" and shutil.which("rocm-smi"):
        smi_command = ["rocm-smi", "--showuse", "--showmemuse"]
        smi_label = "ROCm GPU"
    if smi_command is None:
        return
    try:
        output = subprocess.check_output(
            smi_command,
            text=True,
            timeout=2,
        ).strip()
        logger.debug(
            "%s utilization: %s (epoch=%s batch=%s)",
            smi_label,
            output,
            epoch,
            batch_idx,
        )
    except Exception:
        logger.debug(
            "%s utilization: unavailable (epoch=%s batch=%s)",
            smi_label,
            epoch,
            batch_idx,
        )


def suggest_risk_params_from_model(
    model: CryptoLSTM,
    processor: DataProcessor,
    df: pd.DataFrame,
) -> RiskParams:
    data_scaled, _, df, _ = processor.process_for_training(df, conf.FEATURE_COLS)
    if len(data_scaled) <= conf.SEQ_LENGTH:
        logger = get_logger("bat.training")
        logger.warning("Fallback risk params: insufficient data for model distribution")
        return RiskParams(stop_loss=0.02, take_profit=0.05, max_dd_stop=0.2, position_splits=3)

    # Optimized sampling to prevent OOM
    total_sequences = len(data_scaled) - conf.SEQ_LENGTH
    max_risk_samples = 50000  # Cap samples for risk estimation
    
    if total_sequences > max_risk_samples:
         # Use linspace for uniform coverage of history
         indices = np.linspace(0, total_sequences - 1, max_risk_samples).astype(int)
    else:
         indices = np.arange(total_sequences)

    sequences = []
    for i in indices:
        sequences.append(data_scaled[i : i + conf.SEQ_LENGTH])
    sequences = np.array(sequences)

    model.eval()
    expected_returns = []
    batch_size = 256
    for i in range(0, len(sequences), batch_size):
        batch = sequences[i : i + batch_size]
        x = torch.from_numpy(batch).float().to(conf.DEVICE)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        exp_ret = (probs[:, 2] - probs[:, 0]) * conf.RETURN_THRESHOLD
        expected_returns.extend(exp_ret.tolist())

    expected_returns = np.array(expected_returns)

    lower_q = float(np.quantile(expected_returns, 0.1))
    upper_q = float(np.quantile(expected_returns, 0.9))

    stop_loss = min(max(abs(lower_q), 0.01), 0.1)
    take_profit = min(max(abs(upper_q), 0.02), 0.2)
    max_dd_stop = min(max(stop_loss * 4.0, 0.05), 0.3)

    return RiskParams(
        stop_loss=stop_loss,
        take_profit=take_profit,
        max_dd_stop=max_dd_stop,
        position_splits=3,
    )


def train_model(
    data_path="data/history.csv",
    epochs=None,
    df=None,
    data_pipeline=None,
    research_config: ResearchConfig | None = None,
    model_name=None,
    on_epoch_loss=None,
    on_status=None,
    on_log=None,
    status_every=20,
):
    epochs = epochs or conf.EPOCHS
    print(f"使用裝置: {conf.DEVICE}")
    logger = get_logger("bat.training")
    logger.info("Training start: data_path=%s epochs=%s", data_path, epochs)
    if on_log:
        on_log(f">>> [訓練] 開始 (epochs={epochs})")
    device = conf.DEVICE
    device_backend = getattr(conf, "DEVICE_BACKEND", None)
    if not isinstance(device_backend, str):
        hip_version = getattr(getattr(torch, "version", None), "hip", None)
        device_backend = "rocm" if device.type == "cuda" and hip_version else device.type
    device_name = getattr(conf, "DEVICE_NAME", None)
    if not isinstance(device_name, str):
        device_name = str(device).upper()
    use_amp = device_backend in {"cuda", "rocm"}
    if use_amp:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True
        logger.info("%s enabled: using AMP + torch CUDA API", device_name)
        try:
            requested_fraction = float(os.getenv("BAT_GPU_MEM_FRACTION", "0.5"))
            requested_fraction = max(0.1, min(requested_fraction, 0.9))
            free_mem, total_mem = torch.cuda.mem_get_info()
            available_fraction = free_mem / total_mem if total_mem else 0.0
            total_fraction = min(max(available_fraction * requested_fraction, 0.05), 0.9)
            torch.cuda.set_per_process_memory_fraction(total_fraction)
            logger.info(
                "%s memory fraction set to %.2f (requested=%.2f of available)",
                device_name,
                total_fraction,
                requested_fraction,
            )
            if on_status:
                on_status(
                    {
                        "gpu_mem_fraction": total_fraction,
                        "gpu_mem_available_fraction": requested_fraction,
                        "gpu_mem_free_mb": free_mem / (1024 ** 2),
                        "gpu_mem_total_mb": total_mem / (1024 ** 2),
                    }
                )
        except Exception:
            logger.exception("%s memory limit setup failed; continuing without limit", device_name)

    model_config = _resolve_training_model_config(
        research_config or load_research_config(),
        model_name=model_name,
    )
    inputs = _prepare_training_inputs(
        data_path=data_path,
        df=df,
        data_pipeline=data_pipeline,
        research_config=model_config,
    )
    data_scaled = inputs.data_scaled
    target_scaled = inputs.target_scaled
    df = inputs.df
    target_ret = inputs.target_ret
    processor = inputs.processor
    feature_cols = inputs.feature_cols
    seq_length = inputs.seq_length
    last_trained_timestamp = inputs.last_trained_timestamp
    input_dim = len(feature_cols)
    logger.info(
        "Training data prepared: pipeline=%s samples=%s features=%s seq_length=%s",
        inputs.pipeline,
        len(data_scaled),
        input_dim,
        seq_length,
    )
    logger.debug("Data shape: %s", data_scaled.shape)
    _log_baseline_stats(logger, target_scaled)

    weights = _compute_training_weights(
        target_scaled,
        target_ret,
        class_weight_strength=model_config.training.class_weight_strength,
    )
    logger.info(
        "Sample weights: mean=%.4f min=%.4f max=%.4f",
        float(np.mean(weights)),
        float(np.min(weights)),
        float(np.max(weights)),
    )

    model_name_for_meta = normalize_model_name(model_config.model.name)
    if inputs.pipeline == "research" and model_name_for_meta in SKLEARN_MODEL_NAMES:
        return _train_sklearn_histgb(
            inputs=inputs,
            weights=weights,
            model_config=model_config,
            model_name_for_meta=model_name_for_meta,
            epochs=epochs,
            on_log=on_log,
            on_status=on_status,
        )

    batch_size = int(os.getenv("BAT_BATCH_SIZE", str(conf.BATCH_SIZE)))
    dataset = TimeSeriesDataset(data_scaled, target_scaled, seq_length, weights=weights)
    validation_folds = _validation_folds(len(dataset), model_config)
    if not validation_folds:
        train_indices, valid_indices = _time_ordered_train_valid_indices(
            len(dataset),
            valid_ratio=model_config.validation.holdout_ratio,
        )
        validation_folds = [IndexWalkForwardFold(train_indices, valid_indices)] if train_indices and valid_indices else []
    if not validation_folds:
        raise ValueError("training requires non-empty train and validation splits")
    train_indices = validation_folds[-1].train_indices
    validation_eval_folds = _non_overlapping_validation_folds(train_indices, validation_folds)
    if not validation_eval_folds:
        validation_eval_folds = [validation_folds[-1]]
    train_set = torch.utils.data.Subset(dataset, train_indices)
    future_returns_aligned = _aligned_future_returns(target_ret, seq_length, len(dataset))

    max_full_samples = int(os.getenv("BAT_MAX_FULL_GPU_SAMPLES", "5000"))
    use_full_gpu = use_amp and len(train_set) <= max_full_samples
    use_chunked = use_amp and not use_full_gpu
    if use_amp:
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            samples = len(train_set)
            est_bytes = samples * seq_length * input_dim * 4
            est_bytes += samples * 4
            if est_bytes > free_mem * 0.8:
                use_full_gpu = False
                use_chunked = True
                logger.info(
                    "Force chunked: est=%.2fMB free=%.2fMB total=%.2fMB",
                    est_bytes / (1024 ** 2),
                    free_mem / (1024 ** 2),
                    total_mem / (1024 ** 2),
                )
        except Exception:
            logger.warning("GPU memory check failed, fallback to default scheduling")
    if use_full_gpu:
        x_list = []
        y_list = []
        w_list = []
        for i in range(len(train_set)):
            x_item, y_item, w_item = train_set[i]
            x_list.append(x_item)
            y_list.append(y_item)
            w_list.append(w_item)
        x_all = torch.stack(x_list).to(device)
        y_all = torch.stack(y_list).to(device)
        w_all = torch.stack(w_list).to(device)
        logger.info("Training data moved to GPU: samples=%s", len(train_set))
        train_loader = None
        if on_status:
            on_status({"mode": "full_gpu", "samples": len(train_set)})
        total_batches_per_epoch = 1
    elif use_chunked:
        train_loader = None
        chunk_size = min(20000, len(train_set))
        chunk_stride = max(10000, chunk_size // 2)
        chunk_ranges = []
        for start in range(0, len(train_set), chunk_stride):
            end = min(start + chunk_size, len(train_set))
            if end - start > 0:
                chunk_ranges.append((start, end))
        total_batches_per_epoch = sum(
            (end - start + batch_size - 1) // batch_size
            for start, end in chunk_ranges
        )
        logger.info(
            "Training in chunks: samples=%s chunk_size=%s stride=%s",
            len(train_set),
            chunk_size,
            chunk_stride,
        )
        if on_status:
            on_status(
                {
                    "mode": "chunked",
                    "samples": len(train_set),
                    "chunk_size": chunk_size,
                    "chunk_stride": chunk_stride,
                    "batch_size": batch_size,
                    "batches": total_batches_per_epoch,
                }
            )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=use_amp,
        )
        logger.debug("Train size: %s, Batch size: %s", len(train_set), batch_size)
        if on_status:
            on_status(
                {
                    "mode": "dataloader",
                    "samples": len(train_set),
                    "batch_size": batch_size,
                    "batches": len(train_loader),
                }
            )
        total_batches_per_epoch = len(train_loader)

    if inputs.pipeline == "research":
        model = create_model(model_config, input_dim=input_dim).to(device)
        run_config = model_config
    else:
        model = CryptoLSTM(
            input_dim=input_dim,
            hidden_dim=conf.HIDDEN_SIZE,
            num_layers=conf.NUM_LAYERS,
            dropout=conf.DROPOUT,
        ).to(device)
        model_name_for_meta = "lstm"
        run_config = with_model_name(model_config, "lstm")
    training_run = _create_training_run(
        config=run_config,
        model_name=model_name_for_meta,
        feature_cols=feature_cols,
        targets=target_scaled,
    )
    logger.info("Training run directory: %s", training_run.path)
    if on_log:
        on_log(f">>> [訓練] Run directory: {training_run.path}")
    model_path = "data/lstm_model.pth"
    start_epoch = 0
    if os.path.exists(_checkpoint_path):
        try:
            checkpoint = safe_torch_load(_checkpoint_path, map_location=device)
            meta = checkpoint.get("meta", {})
            if (
                meta.get("output_dim") == 3
                and meta.get("input_dim") == input_dim
                and meta.get("model_name", "lstm") == model_name_for_meta
            ):
                model.load_state_dict(checkpoint["model_state"])
                logger.info("Loaded checkpoint for resume: %s", _checkpoint_path)
                start_epoch = int(checkpoint.get("epoch", 0))
            else:
                logger.warning("Checkpoint incompatible, ignoring: %s", _checkpoint_path)
        except Exception:
            logger.exception("Failed to load checkpoint, fallback to fresh weights")
    elif os.path.exists(model_path):
        try:
            state = load_manifested_torch_state(model_path, map_location=device)
            model.load_state_dict(state)
            logger.info("Loaded existing model for incremental training: %s", model_path)
        except ArtifactSecurityError:
            logger.exception("Existing model artifact rejected, fallback to fresh weights")
        except Exception:
            logger.exception("Failed to load existing model, fallback to fresh weights")
    logger.debug(
        "Model config: input_dim=%s hidden_dim=%s layers=%s dropout=%.3f lr=%.6f",
        input_dim,
        conf.HIDDEN_SIZE,
        conf.NUM_LAYERS,
        conf.DROPOUT,
        conf.LR,
    )

    logger.debug("Last trained timestamp identified: %s", last_trained_timestamp)

    criterion = nn.CrossEntropyLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=conf.LR)
    if start_epoch and os.path.exists(_checkpoint_path):
        try:
            optimizer.load_state_dict(checkpoint.get("optimizer_state", {}))
            logger.info("Loaded optimizer state from checkpoint")
        except Exception:
            logger.exception("Failed to load optimizer state, resetting optimizer")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    print(">>> 開始訓練...")
    model.train()
    target_epoch = start_epoch + epochs
    early_stop_state = None
    completed_epochs = start_epoch
    stopped_early = False
    stopped_by_user = False
    latest_epoch_metrics: dict = {}
    for epoch in range(start_epoch, target_epoch):
        if should_stop_training():
            stopped_by_user = True
            logger.warning("Training stopped by user")
            break
        if on_status:
            on_status({"epoch": epoch + 1, "epochs": target_epoch})
        total_loss = 0.0
        if use_full_gpu:
            if should_stop_training():
                stopped_by_user = True
                logger.warning("Training stopped by user (full_gpu)")
                break
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(x_all)
                loss = criterion(output, y_all.squeeze(1))
                loss = (loss * w_all.squeeze(1)).mean()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss = loss.item()
            logger.debug(
                "Epoch %s/%s FullBatch Loss=%.8f",
                epoch + 1,
                target_epoch,
                loss.item(),
            )
            if use_amp:
                _log_gpu_stats(logger, epoch + 1, 1)
            batch_count = 1
            if on_status:
                on_status({"batch": 1, "batches": 1, "progress": 1.0})
        elif use_chunked:
            total_batches = 0
            for chunk_index, (start, end) in enumerate(chunk_ranges, start=1):
                if should_stop_training():
                    stopped_by_user = True
                    logger.warning("Training stopped by user (chunked)")
                    break
                x_list = []
                y_list = []
                w_list = []
                for i in range(start, end):
                    x_item, y_item, w_item = train_set[i]
                    x_list.append(x_item)
                    y_list.append(y_item)
                    w_list.append(w_item)
                x_chunk = torch.stack(x_list).to(device)
                y_chunk = torch.stack(y_list).to(device)
                w_chunk = torch.stack(w_list).to(device)
                for batch_start in range(0, len(x_chunk), batch_size):
                    if should_stop_training():
                        stopped_by_user = True
                        logger.warning("Training stopped by user (chunked batch)")
                        break
                    batch_end = batch_start + batch_size
                    x_batch = x_chunk[batch_start:batch_end]
                    y_batch = y_chunk[batch_start:batch_end]
                    w_batch = w_chunk[batch_start:batch_end]
                    optimizer.zero_grad()
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        output = model(x_batch)
                        loss = criterion(output, y_batch.squeeze(1))
                        loss = (loss * w_batch.squeeze(1)).mean()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    total_loss += loss.item()
                    total_batches += 1
                    if on_status and total_batches % status_every == 0:
                        on_status(
                            {
                                "batch": total_batches,
                                "batches": total_batches_per_epoch,
                                "progress": total_batches / max(total_batches_per_epoch, 1),
                                "chunk_index": chunk_index,
                                "chunks": len(chunk_ranges),
                                "chunk_progress": (batch_start + len(x_batch)) / max(len(x_chunk), 1),
                            }
                        )
                    logger.debug(
                        "Epoch %s Chunk %s-%s Batch %s Loss=%.8f",
                        epoch + 1,
                        start,
                        end,
                        batch_start // batch_size + 1,
                        loss.item(),
                    )
                    if use_amp and total_batches % 20 == 0:
                        _log_gpu_stats(logger, epoch + 1, total_batches)
                del x_chunk, y_chunk, w_chunk
                if use_amp:
                    torch.cuda.empty_cache()
            batch_count = max(total_batches, 1)
        else:
            for batch_idx, (x_batch, y_batch, w_batch) in enumerate(train_loader, start=1):
                if should_stop_training():
                    stopped_by_user = True
                    logger.warning("Training stopped by user (dataloader)")
                    break
                x_batch = x_batch.to(device, non_blocking=use_amp)
                y_batch = y_batch.to(device, non_blocking=use_amp)
                w_batch = w_batch.to(device, non_blocking=use_amp)

                optimizer.zero_grad()
                with torch.amp.autocast("cuda", enabled=use_amp):
                    output = model(x_batch)
                    loss = criterion(output, y_batch.squeeze(1))
                    loss = (loss * w_batch.squeeze(1)).mean()
                scaler.scale(loss).backward()
                total_grad_norm = 0.0
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = float(param.grad.data.norm(2).item())
                        total_grad_norm += grad_norm
                        logger.debug(
                            "Epoch %s Batch %s GradNorm %s=%.8f",
                            epoch + 1,
                            batch_idx,
                            name,
                            grad_norm,
                        )
                logger.debug(
                    "Epoch %s Batch %s TotalGradNorm=%.8f",
                    epoch + 1,
                    batch_idx,
                    total_grad_norm,
                )
                for name, param in model.named_parameters():
                    param_data = param.data
                    logger.debug(
                        "Epoch %s Batch %s ParamStats %s mean=%.8f std=%.8f min=%.8f max=%.8f",
                        epoch + 1,
                        batch_idx,
                        name,
                        float(param_data.mean().item()),
                        float(param_data.std().item()),
                        float(param_data.min().item()),
                        float(param_data.max().item()),
                    )
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
                logger.debug(
                    "Epoch %s/%s Batch %s/%s Loss=%.8f",
                    epoch + 1,
                    target_epoch,
                    batch_idx,
                    len(train_loader),
                    loss.item(),
                )
                if use_amp and batch_idx % 20 == 0:
                    _log_gpu_stats(logger, epoch + 1, batch_idx)
                if on_status and batch_idx % status_every == 0:
                    on_status(
                        {
                            "batch": batch_idx,
                            "batches": len(train_loader),
                            "progress": batch_idx / max(len(train_loader), 1),
                            "chunk_index": 1,
                            "chunks": 1,
                            "chunk_progress": batch_idx / max(len(train_loader), 1),
                        }
                    )
            batch_count = max(len(train_loader), 1)

        if stopped_by_user:
            break

        avg_loss = total_loss / max(batch_count, 1)
        fold_metrics = [
            evaluate_model_on_dataset(
                model=model,
                dataset=dataset,
                indices=fold.valid_indices,
                device=device,
                fee=model_config.label.fee,
                slippage=model_config.label.slippage,
                future_returns=future_returns_aligned,
                batch_size=batch_size,
                confidence_threshold=model_config.validation.decision_confidence_threshold,
                edge_threshold=model_config.validation.decision_edge_threshold,
            )
            for fold in validation_eval_folds
        ]
        validation_metrics = _aggregate_fold_metrics(fold_metrics)
        validation_metrics["validation_method"] = str(model_config.validation.method)
        epoch_metrics = {"epoch": epoch + 1, "train_loss": avg_loss, **validation_metrics}
        latest_epoch_metrics = epoch_metrics
        training_run.write_epoch_metrics(epoch_metrics)
        completed_epochs = epoch + 1
        early_stop_state = _update_early_stopping(
            early_stop_state,
            epoch_metrics,
            epoch=epoch + 1,
            metric_name=model_config.training.early_stopping_metric,
            patience=model_config.training.early_stopping_patience,
        )
        if early_stop_state.improved:
            _save_run_best_model(training_run, model)
        stopped_early = early_stop_state.should_stop
        if on_status:
            status_payload = {
                "batch": batch_count,
                "batches": max(total_batches_per_epoch, 1),
                "progress": 1.0,
                "chunk_index": 1 if not use_chunked else len(chunk_ranges),
                "chunks": 1 if not use_chunked else len(chunk_ranges),
                "chunk_progress": 1.0,
            }
            status_payload.update(validation_metrics)
            if stopped_early:
                status_payload.update(
                    {
                        "early_stopped": True,
                        "early_stop_reason": early_stop_state.stop_reason,
                        "early_stopping_metric": early_stop_state.metric_name,
                        "best_epoch": early_stop_state.best_epoch,
                        "best_metric": early_stop_state.best_metric,
                    }
                )
            on_status(status_payload)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{target_epoch}, Loss: {avg_loss:.6f}")
            logger.debug("Epoch %s avg loss=%.8f", epoch + 1, avg_loss)
        logger.info("Epoch %s/%s avg loss=%.8f", epoch + 1, target_epoch, avg_loss)
        if on_log:
            try:
                message = f">>> [訓練] Epoch {epoch + 1}/{target_epoch} loss={avg_loss:.6f}"
                if validation_metrics:
                    message += (
                        f" val_loss={validation_metrics['val_loss']:.6f}"
                        f" val_macro_f1={validation_metrics['val_macro_f1']:.4f}"
                        f" expected_ret={validation_metrics['expected_return_after_cost']:.6f}"
                    )
                on_log(message)
                if stopped_early:
                    on_log(f">>> [訓練] Early stopping: {early_stop_state.stop_reason}")
            except Exception: pass
            
        if on_epoch_loss:
            # fix(error): prevent callback failure from crashing training
            try:
                on_epoch_loss(avg_loss)
            except Exception:
                logger.error("Callback on_epoch_loss failed")
            
        # Save checkpoint every epoch for safety
        try:
            os.makedirs(os.path.dirname(_checkpoint_path), exist_ok=True)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                        "meta": {
                            "input_dim": input_dim,
                            "output_dim": 3,
                            "model_name": model_name_for_meta,
                            "loss": avg_loss,
                            "last_trained_timestamp": last_trained_timestamp
                    },
                },
                _checkpoint_path,
            )
            # Optional: Also update the main model path if it's the best loss? 
            # For now, checkpoint is enough to resume.
        except Exception:
            logger.exception("Failed to save training checkpoint")

        if stopped_early:
            logger.warning("Early stopping: %s", early_stop_state.stop_reason)
            break

    if processor is not None:
        _log_return_alignment(logger, model, processor, data_scaled, df)
    admission = assess_model_admission(latest_epoch_metrics, [_run_artifact_path(training_run, "best_model.pt")])
    if admission.passed:
        torch.save(model.state_dict(), model_path)
        write_artifact_manifest(
            Path(model_path).parent,
            [
                {
                    "path": Path(model_path),
                    "type": "torch_state_dict",
                    "runtime_load_allowed": True,
                }
            ],
            feature_columns=feature_cols,
            config_sha256=None,
            training_data_sha256=None,
            git_sha=None,
            admission=admission.to_dict(),
        )
        print(f">>> 模型已保存至 {model_path}")
        logger.info("Model saved: %s", model_path)
        if on_log:
            on_log(f">>> [訓練] 模型已保存 {model_path}")
    else:
        logger.warning("Model not promoted: admission failed %s", admission.reasons)
        if on_log:
            on_log(f">>> [訓練] Model not promoted: {', '.join(admission.reasons)}")
    if hasattr(training_run, "write_report"):
        report_status = "completed"
        if stopped_by_user:
            report_status = "stopped"
        elif stopped_early:
            report_status = "early_stopped"
        stop_reason = None
        if stopped_by_user:
            stop_reason = "stopped by user"
        elif early_stop_state:
            stop_reason = early_stop_state.stop_reason
        training_run.write_report(
            {
                "status": report_status,
                "monitored_metric": model_config.training.early_stopping_metric,
                "patience": model_config.training.early_stopping_patience,
                "completed_epochs": completed_epochs,
                "best_epoch": early_stop_state.best_epoch if early_stop_state else 0,
                "best_metric": early_stop_state.best_metric if early_stop_state else None,
                "stop_reason": stop_reason,
                "validation_method": str(model_config.validation.method),
                "fold_count": latest_epoch_metrics.get("fold_count", 0),
                "admission": admission.to_dict(),
            }
        )
    return model, processor, df, last_trained_timestamp


def train_and_backtest(
    data_path="data/history.csv",
    epochs=None,
    data_pipeline=None,
    research_config: ResearchConfig | None = None,
    model_name=None,
    on_epoch_loss=None,
    on_status=None,
    on_log=None,
    status_every=20,
):
    config = _resolve_training_model_config(
        research_config or load_research_config(),
        model_name=model_name,
    )
    selected_pipeline = _normalize_data_pipeline(
        data_pipeline if data_pipeline is not None else config.training.data_pipeline
    )
    df = None if selected_pipeline == "research" and config.training.use_preprocessing_cache else _load_training_data(data_path)
    model, processor, df, last_trained_ts = train_model(
        data_path=data_path,
        epochs=epochs,
        df=df,
        data_pipeline=selected_pipeline,
        research_config=config,
        model_name=config.model.name,
        on_epoch_loss=on_epoch_loss,
        on_status=on_status,
        on_log=on_log,
        status_every=status_every,
    )

    if processor is None:
        risk = RiskParams(
            stop_loss=config.risk.stop_loss,
            take_profit=config.risk.take_profit,
            max_dd_stop=config.risk.max_drawdown_stop,
            position_splits=3,
        )
    else:
        risk = suggest_risk_params_from_model(model, processor, df)
    logger = get_logger("bat.training")
    logger.info(
        "Risk params suggested: stop_loss=%.4f take_profit=%.4f max_dd=%.4f splits=%s",
        risk.stop_loss,
        risk.take_profit,
        risk.max_dd_stop,
        risk.position_splits,
    )
    result = run_backtest(
        data_path=data_path,
        fee=config.backtest.taker_fee,
        slippage=config.backtest.slippage,
        position_splits=risk.position_splits,
        stop_loss_pct=risk.stop_loss,
        take_profit_pct=risk.take_profit,
        max_drawdown_stop=risk.max_dd_stop,
        return_equity=True,
    )

    return result, risk, last_trained_ts
