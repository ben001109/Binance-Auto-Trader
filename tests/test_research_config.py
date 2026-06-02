import tempfile
import unittest
from pathlib import Path

from bat.research_config import load_research_config


class ResearchConfigTest(unittest.TestCase):
    def test_loads_yaml_into_typed_config(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "training.yaml"
            path.write_text(
                """
data:
  symbol: ETHUSDT
  interval: 1h
  seq_len: 48
  horizon: 6
  rolling_norm_window: 120
label:
  fee: 0.001
  slippage: 0.0005
  min_edge: 0.004
model:
  name: cnn_lstm
risk:
  paper_only: true
  trading_enabled: false
""".strip(),
                encoding="utf-8",
            )

            config = load_research_config(path)

        self.assertEqual(config.data.symbol, "ETHUSDT")
        self.assertEqual(config.data.interval, "1h")
        self.assertEqual(config.data.seq_len, 48)
        self.assertEqual(config.data.horizon, 6)
        self.assertEqual(config.label.fee, 0.001)
        self.assertEqual(config.model.name, "cnn_lstm")
        self.assertTrue(config.risk.paper_only)
        self.assertFalse(config.risk.trading_enabled)

    def test_default_config_is_paper_only(self):
        config = load_research_config()

        self.assertTrue(config.risk.paper_only)
        self.assertFalse(config.risk.trading_enabled)


if __name__ == "__main__":
    unittest.main()
