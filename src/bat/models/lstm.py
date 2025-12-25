import torch.nn as nn

class CryptoLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim=3, dropout=0.2):
        super(CryptoLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout
        )
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x shape: (batch, seq, feature)
        out, _ = self.lstm(x)
        # 取最後一個時間步
        out = out[:, -1, :]
        out = self.fc(out)
        return out
