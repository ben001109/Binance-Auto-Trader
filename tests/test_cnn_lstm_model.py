import unittest

import torch

from bat.models import CryptoCNNLSTM


class CnnLstmModelTest(unittest.TestCase):
    def test_forward_returns_classification_logits(self):
        model = CryptoCNNLSTM(
            input_dim=14,
            seq_len=60,
            cnn_channels=32,
            kernel_size=5,
            lstm_hidden=64,
            lstm_layers=2,
            dropout=0.2,
            output_dim=3,
        )
        x = torch.randn(4, 60, 14)

        logits = model(x)

        self.assertEqual(tuple(logits.shape), (4, 3))

    def test_constructor_parameters_configure_layers(self):
        model = CryptoCNNLSTM(
            input_dim=7,
            seq_len=30,
            cnn_channels=16,
            kernel_size=3,
            lstm_hidden=24,
            lstm_layers=1,
            dropout=0.1,
            output_dim=3,
        )

        self.assertEqual(model.conv.in_channels, 7)
        self.assertEqual(model.conv.out_channels, 16)
        self.assertEqual(model.conv.kernel_size, (3,))
        self.assertEqual(model.lstm.input_size, 16)
        self.assertEqual(model.lstm.hidden_size, 24)
        self.assertEqual(model.classifier.out_features, 3)


if __name__ == "__main__":
    unittest.main()
