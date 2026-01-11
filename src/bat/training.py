import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import subprocess
import shutil
import os

from dataclasses import dataclass

from bat.config import conf
from bat.models.lstm import CryptoLSTM
from bat.data.dataset import DataProcessor, TimeSeriesDataset
from bat.backtest import run_backtest
import numpy as np
from bat.logger import get_logger


@dataclass
class RiskParams:
    stop_loss: float
    take_profit: float
    max_dd_stop: float
    position_splits: int


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
    if not torch.cuda.is_available():
        return
    allocated = torch.cuda.memory_allocated() / (1024 ** 2)
    reserved = torch.cuda.memory_reserved() / (1024 ** 2)
    logger.debug(
        "GPU memory: allocated=%.2fMB reserved=%.2fMB (epoch=%s batch=%s)",
        allocated,
        reserved,
        epoch,
        batch_idx,
    )
    if shutil.which("nvidia-smi") is None:
        return
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
        ).strip()
        logger.debug(
            "GPU utilization: %s (epoch=%s batch=%s)",
            output,
            epoch,
            batch_idx,
        )
    except Exception:
        logger.debug("GPU utilization: unavailable (epoch=%s batch=%s)", epoch, batch_idx)


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
    use_amp = device.type == "cuda"
    if use_amp:
        torch.backends.cudnn.benchmark = True
        logger.info("CUDA enabled: using AMP + cuDNN benchmark")
        requested_fraction = float(os.getenv("BAT_GPU_MEM_FRACTION", "0.5"))
        requested_fraction = max(0.1, min(requested_fraction, 0.9))
        free_mem, total_mem = torch.cuda.mem_get_info()
        available_fraction = free_mem / total_mem if total_mem else 0.0
        total_fraction = min(max(available_fraction * requested_fraction, 0.05), 0.9)
        torch.cuda.set_per_process_memory_fraction(total_fraction)
        logger.info(
            "CUDA memory fraction set to %.2f (requested=%.2f of available)",
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

    df = df if df is not None else _load_training_data(data_path)
    
    # Identify last available timestamp BEFORE processing (which drops tail rows)
    last_trained_timestamp = 0
    if "timestamp" in df.columns:
        # Check if it's already datetime (if passed in as df)
        if pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            # If it's already datetime, convert to ms
            val = df["timestamp"].max()
            if pd.notnull(val):
                last_trained_timestamp = int(val.value // 1_000_000)
        else:
             # Assume int ms
             val = df["timestamp"].max()
             if pd.notnull(val):
                 last_trained_timestamp = int(val)
    elif "close_time" in df.columns:
         val = df["close_time"].max()
         if pd.notnull(val):
             last_trained_timestamp = int(val)
             
    processor = DataProcessor()

    data_scaled, target_scaled, df, target_ret = processor.process_for_training(df, conf.FEATURE_COLS)
    logger.debug("Data shape: %s", data_scaled.shape)
    _log_baseline_stats(logger, target_scaled)

    class_counts = np.bincount(target_scaled, minlength=3).astype(float)
    class_counts[class_counts == 0] = 1.0
    class_weights = class_counts.sum() / (3.0 * class_counts)
    abs_ret = np.abs(target_ret)
    scale = np.percentile(abs_ret, 90) if len(abs_ret) else 0.0
    if scale <= 0:
        sample_weights = np.ones_like(abs_ret, dtype=float)
    else:
        sample_weights = 1.0 + np.clip(abs_ret / scale, 0.0, 3.0)
    weights = sample_weights * class_weights[target_scaled]
    logger.info(
        "Sample weights: mean=%.4f min=%.4f max=%.4f",
        float(np.mean(weights)),
        float(np.min(weights)),
        float(np.max(weights)),
    )

    batch_size = int(os.getenv("BAT_BATCH_SIZE", str(conf.BATCH_SIZE)))
    dataset = TimeSeriesDataset(data_scaled, target_scaled, conf.SEQ_LENGTH, weights=weights)
    train_size = int(len(dataset) * 0.8)
    train_indices = list(range(train_size))
    train_set = torch.utils.data.Subset(dataset, train_indices)

    max_full_samples = int(os.getenv("BAT_MAX_FULL_GPU_SAMPLES", "5000"))
    use_full_gpu = use_amp and len(train_set) <= max_full_samples
    use_chunked = use_amp and not use_full_gpu
    if use_amp:
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            samples = len(train_set)
            est_bytes = samples * conf.SEQ_LENGTH * len(conf.FEATURE_COLS) * 4
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

    model = CryptoLSTM(
        input_dim=len(conf.FEATURE_COLS),
        hidden_dim=conf.HIDDEN_SIZE,
        num_layers=conf.NUM_LAYERS,
        dropout=conf.DROPOUT
    ).to(device)
    model_path = "data/lstm_model.pth"
    start_epoch = 0
    if os.path.exists(_checkpoint_path):
        try:
            checkpoint = torch.load(_checkpoint_path, map_location=device)
            meta = checkpoint.get("meta", {})
            if meta.get("output_dim") == 3 and meta.get("input_dim") == len(conf.FEATURE_COLS):
                model.load_state_dict(checkpoint["model_state"])
                logger.info("Loaded checkpoint for resume: %s", _checkpoint_path)
                start_epoch = int(checkpoint.get("epoch", 0))
            else:
                logger.warning("Checkpoint incompatible, ignoring: %s", _checkpoint_path)
        except Exception:
            logger.exception("Failed to load checkpoint, fallback to fresh weights")
    elif os.path.exists(model_path):
        try:
            model.load_state_dict(torch.load(model_path, map_location=device))
            logger.info("Loaded existing model for incremental training: %s", model_path)
        except Exception:
            logger.exception("Failed to load existing model, fallback to fresh weights")
    logger.debug(
        "Model config: input_dim=%s hidden_dim=%s layers=%s dropout=%.3f lr=%.6f",
        len(conf.FEATURE_COLS),
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
    for epoch in range(start_epoch, target_epoch):
        if should_stop_training():
            logger.warning("Training stopped by user")
            break
        if on_status:
            on_status({"epoch": epoch + 1, "epochs": target_epoch})
        total_loss = 0.0
        if use_full_gpu:
            if should_stop_training():
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

        avg_loss = total_loss / max(batch_count, 1)
        if on_status:
            on_status(
                {
                    "batch": batch_count,
                    "batches": max(total_batches_per_epoch, 1),
                    "progress": 1.0,
                    "chunk_index": 1 if not use_chunked else len(chunk_ranges),
                    "chunks": 1 if not use_chunked else len(chunk_ranges),
                    "chunk_progress": 1.0,
                }
            )
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1}/{target_epoch}, Loss: {avg_loss:.6f}")
            logger.debug("Epoch %s avg loss=%.8f", epoch + 1, avg_loss)
        logger.info("Epoch %s/%s avg loss=%.8f", epoch + 1, target_epoch, avg_loss)
        if on_log:
            try:
                on_log(f">>> [訓練] Epoch {epoch + 1}/{target_epoch} loss={avg_loss:.6f}")
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
                        "input_dim": len(conf.FEATURE_COLS),
                        "output_dim": 3,
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

    _log_return_alignment(logger, model, processor, data_scaled, df)
    torch.save(model.state_dict(), model_path)
    print(f">>> 模型已保存至 {model_path}")
    logger.info("Model saved: %s", model_path)
    if on_log:
        on_log(f">>> [訓練] 模型已保存 {model_path}")
    return model, processor, df, last_trained_timestamp


def train_and_backtest(
    data_path="data/history.csv",
    epochs=None,
    on_epoch_loss=None,
    on_status=None,
    on_log=None,
    status_every=20,
):
    df = _load_training_data(data_path)
    model, processor, df, last_trained_ts = train_model(
        data_path=data_path,
        epochs=epochs,
        df=df,
        on_epoch_loss=on_epoch_loss,
        on_status=on_status,
        on_log=on_log,
        status_every=status_every,
    )

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
        fee=0.001,
        slippage=0.0005,
        position_splits=risk.position_splits,
        stop_loss_pct=risk.stop_loss,
        take_profit_pct=risk.take_profit,
        max_drawdown_stop=risk.max_dd_stop,
        return_equity=True,
    )

    return result, risk, last_trained_ts
