from __future__ import annotations

from bat.models.cnn_lstm import CryptoCNNLSTM
from bat.models.lstm import CryptoLSTM
from bat.research_config import ResearchConfig


def normalize_model_name(name: str | None) -> str:
    normalized = str(name or "lstm").strip().lower().replace("-", "_")
    if normalized in {"lstm", "cnn_lstm"}:
        return normalized
    if normalized in {"histgb", "hist_gb", "hist_gradient_boosting", "hist_gradient_boosting_classifier"}:
        return "histgb"
    raise ValueError(f"Unsupported model name: {name}")


def create_model(config: ResearchConfig, input_dim: int):
    model_name = normalize_model_name(config.model.name)
    if model_name == "lstm":
        return CryptoLSTM(
            input_dim=input_dim,
            hidden_dim=config.model.lstm_hidden,
            num_layers=config.model.lstm_layers,
            output_dim=config.model.output_dim,
            dropout=config.model.dropout if config.model.lstm_layers > 1 else 0.0,
        )
    if model_name == "cnn_lstm":
        return CryptoCNNLSTM(
            input_dim=input_dim,
            seq_len=config.data.seq_len,
            cnn_channels=config.model.cnn_channels,
            kernel_size=config.model.kernel_size,
            lstm_hidden=config.model.lstm_hidden,
            lstm_layers=config.model.lstm_layers,
            dropout=config.model.dropout,
            output_dim=config.model.output_dim,
        )
    raise ValueError(f"Unsupported model name: {config.model.name}")
