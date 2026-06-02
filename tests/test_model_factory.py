import unittest

from bat.models import CryptoCNNLSTM, CryptoLSTM
from bat.models.factory import create_model, normalize_model_name
from bat.research_config import DataConfig, ModelConfig, ResearchConfig


class ModelFactoryTest(unittest.TestCase):
    def test_create_lstm_model(self):
        config = ResearchConfig(
            model=ModelConfig(name="lstm", lstm_hidden=32, lstm_layers=1, dropout=0.1),
        )

        model = create_model(config, input_dim=7)

        self.assertIsInstance(model, CryptoLSTM)

    def test_create_cnn_lstm_model(self):
        config = ResearchConfig(
            data=DataConfig(seq_len=30),
            model=ModelConfig(
                name="cnn_lstm",
                cnn_channels=16,
                kernel_size=3,
                lstm_hidden=24,
                lstm_layers=1,
                dropout=0.1,
            ),
        )

        model = create_model(config, input_dim=7)

        self.assertIsInstance(model, CryptoCNNLSTM)
        self.assertEqual(model.conv.in_channels, 7)
        self.assertEqual(model.seq_len, 30)

    def test_unknown_model_name_raises(self):
        config = ResearchConfig(model=ModelConfig(name="transformer"))

        with self.assertRaises(ValueError):
            create_model(config, input_dim=7)

    def test_hist_gradient_boosting_alias_normalizes_to_histgb(self):
        self.assertEqual(normalize_model_name("hist_gradient_boosting"), "histgb")
        self.assertEqual(normalize_model_name("hist-gb"), "histgb")


if __name__ == "__main__":
    unittest.main()
