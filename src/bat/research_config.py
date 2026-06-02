from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, TypeVar

import yaml

from bat.risk.live_state import validate_live_state


DEFAULT_CONFIG_PATH = Path("configs/training.yaml")


@dataclass(frozen=True)
class DataConfig:
    symbol: str = "BTCUSDT"
    interval: str = "15m"
    seq_len: int = 60
    horizon: int = 4
    rolling_norm_window: int = 96


@dataclass(frozen=True)
class FeatureConfig:
    use_log_returns: bool = True
    use_rsi: bool = True
    use_macd: bool = True
    use_atr: bool = True
    use_bollinger_width: bool = True
    use_volume_features: bool = True


@dataclass(frozen=True)
class LabelConfig:
    mode: str = "classification"
    fee: float = 0.001
    slippage: float = 0.0005
    min_edge: float = 0.003


@dataclass(frozen=True)
class ModelConfig:
    name: str = "cnn_lstm"
    cnn_channels: int = 64
    kernel_size: int = 5
    lstm_hidden: int = 128
    lstm_layers: int = 2
    dropout: float = 0.3
    output_dim: int = 3


@dataclass(frozen=True)
class TrainingConfig:
    data_pipeline: str = "legacy"
    use_preprocessing_cache: bool = True
    epochs: int = 50
    batch_size: int = 128
    lr: float = 0.0001
    early_stopping_patience: int = 8
    early_stopping_metric: str = "expected_return_after_cost"
    class_weight_strength: float = 1.0


@dataclass(frozen=True)
class ValidationConfig:
    method: str = "walk_forward"
    holdout_ratio: float = 0.2
    decision_confidence_threshold: float = 0.55
    decision_edge_threshold: float = 0.0
    train_window: str = "90d"
    valid_window: str = "14d"
    step_window: str = "14d"


@dataclass(frozen=True)
class BacktestConfig:
    initial_capital: float = 10000.0
    maker_fee: float = 0.001
    taker_fee: float = 0.001
    slippage: float = 0.0005


@dataclass(frozen=True)
class RiskConfig:
    max_drawdown_stop: float = 0.2
    max_position_fraction: float = 0.25
    stop_loss: float = 0.03
    take_profit: float = 0.06
    cooldown_after_loss: int = 3
    live_state: str = "PAPER"
    max_order_equity_fraction: float = 0.01
    max_daily_loss_fraction: float = 0.005
    max_consecutive_losses: int = 2
    cooldown_minutes: int = 60
    paper_only: bool = True
    trading_enabled: bool = False

    def __post_init__(self) -> None:
        validate_live_state(self.live_state)


@dataclass(frozen=True)
class ResearchConfig:
    data: DataConfig = DataConfig()
    features: FeatureConfig = FeatureConfig()
    label: LabelConfig = LabelConfig()
    model: ModelConfig = ModelConfig()
    training: TrainingConfig = TrainingConfig()
    validation: ValidationConfig = ValidationConfig()
    backtest: BacktestConfig = BacktestConfig()
    risk: RiskConfig = RiskConfig()


T = TypeVar("T")


def _build_dataclass(cls: type[T], values: dict[str, Any] | None) -> T:
    allowed = {field.name for field in fields(cls)}
    payload = {key: value for key, value in (values or {}).items() if key in allowed}
    return cls(**payload)


def load_research_config(path: str | Path | None = None) -> ResearchConfig:
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    raw: dict[str, Any] = {}
    if config_path.exists():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            raw = loaded

    return ResearchConfig(
        data=_build_dataclass(DataConfig, raw.get("data")),
        features=_build_dataclass(FeatureConfig, raw.get("features")),
        label=_build_dataclass(LabelConfig, raw.get("label")),
        model=_build_dataclass(ModelConfig, raw.get("model")),
        training=_build_dataclass(TrainingConfig, raw.get("training")),
        validation=_build_dataclass(ValidationConfig, raw.get("validation")),
        backtest=_build_dataclass(BacktestConfig, raw.get("backtest")),
        risk=_build_dataclass(RiskConfig, raw.get("risk")),
    )


def with_model_name(config: ResearchConfig, model_name: str | None) -> ResearchConfig:
    if not model_name:
        return config
    normalized = str(model_name).strip().lower().replace("-", "_")
    return replace(config, model=replace(config.model, name=normalized))
