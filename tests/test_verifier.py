import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from bat.verifier import Verifier


class VerifierTest(unittest.IsolatedAsyncioTestCase):
    async def test_verify_analyst_injects_offline_client(self):
        fake_agent = MagicMock()
        fake_agent.strategy = SimpleNamespace(training_data_path=None)
        fake_agent.analyze = AsyncMock(
            return_value=(
                SimpleNamespace(action="HOLD", confidence=1.0),
                SimpleNamespace(stop_loss=0.01, take_profit=0.02),
            )
        )

        with patch("bat.verifier.AnalystAgent", return_value=fake_agent) as agent_cls:
            await Verifier.verify_analyst()

        self.assertIsNotNone(agent_cls.call_args.kwargs["client"])


if __name__ == "__main__":
    unittest.main()
