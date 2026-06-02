import torch
import torch.nn as nn


class CryptoCNNLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        seq_len: int,
        cnn_channels: int,
        kernel_size: int,
        lstm_hidden: int,
        lstm_layers: int,
        dropout: float,
        output_dim: int = 3,
    ):
        super().__init__()
        if input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if cnn_channels <= 0:
            raise ValueError("cnn_channels must be positive")
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if lstm_hidden <= 0:
            raise ValueError("lstm_hidden must be positive")
        if lstm_layers <= 0:
            raise ValueError("lstm_layers must be positive")

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=cnn_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.batch_norm = nn.BatchNorm1d(cnn_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.classifier = nn.Linear(lstm_hidden, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("expected input shape [batch, seq_len, features]")
        if x.shape[-1] != self.input_dim:
            raise ValueError(f"expected input_dim={self.input_dim}, got {x.shape[-1]}")

        x = x.transpose(1, 2)
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :])
