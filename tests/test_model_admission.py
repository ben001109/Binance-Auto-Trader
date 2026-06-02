import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from bat.services.artifact_security import ArtifactSecurityError, load_manifested_torch_state, sha256_file
from bat.services.model_admission import AdmissionCriteria, assess_model_admission


def good_metrics() -> dict:
    return {
        "fold_count": 3,
        "expected_return_after_cost": 0.002,
        "profit_factor": 1.4,
        "max_drawdown": -0.05,
        "trade_count": 80,
        "prediction_distribution": {0: 20, 1: 40, 2: 20},
    }


class ModelAdmissionTest(unittest.TestCase):
    def test_admission_passes_when_metrics_and_artifact_meet_thresholds(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact = Path(tmpdir) / "best_model.pt"
            artifact.write_bytes(b"model")

            decision = assess_model_admission(good_metrics(), [artifact], AdmissionCriteria())

        self.assertTrue(decision.passed)
        self.assertEqual(decision.reasons, [])

    def test_admission_rejects_sparse_lucky_trade_count(self):
        metrics = good_metrics()
        metrics["trade_count"] = 5

        decision = assess_model_admission(metrics, [], AdmissionCriteria(min_trade_count=30))

        self.assertFalse(decision.passed)
        self.assertIn("trade_count below minimum", decision.reasons)

    def test_admission_rejects_bad_drawdown_profit_or_expected_return(self):
        metrics = good_metrics()
        metrics["expected_return_after_cost"] = -0.001
        metrics["profit_factor"] = 0.8
        metrics["max_drawdown"] = -0.35

        decision = assess_model_admission(metrics, [], AdmissionCriteria())

        self.assertFalse(decision.passed)
        self.assertIn("expected_return_after_cost below minimum", decision.reasons)
        self.assertIn("profit_factor below minimum", decision.reasons)
        self.assertIn("max_drawdown below floor", decision.reasons)

    def test_admission_rejects_non_finite_metrics(self):
        metrics = good_metrics()
        metrics["expected_return_after_cost"] = math.nan

        decision = assess_model_admission(metrics, [], AdmissionCriteria())

        self.assertFalse(decision.passed)
        self.assertIn("expected_return_after_cost unavailable", decision.reasons)

    def test_admission_rejects_concentrated_prediction_distribution(self):
        metrics = good_metrics()
        metrics["prediction_distribution"] = {0: 0, 1: 99, 2: 1}

        decision = assess_model_admission(metrics, [], AdmissionCriteria())

        self.assertFalse(decision.passed)
        self.assertIn("prediction distribution too concentrated", decision.reasons)

    def test_admission_rejects_missing_required_artifact(self):
        decision = assess_model_admission(
            good_metrics(),
            [Path("missing-model.pt")],
            AdmissionCriteria(),
        )

        self.assertFalse(decision.passed)
        self.assertIn("required artifact missing", decision.reasons)

    def test_runtime_loader_rejects_manifest_without_passing_admission(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact = root / "best_model.pt"
            torch.save({"weight": torch.ones(1)}, artifact)
            (root / "manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "artifacts": {
                            artifact.name: {
                                "sha256": sha256_file(artifact),
                                "runtime_load_allowed": True,
                                "type": "torch_state_dict",
                            }
                        },
                    }
                ),
                encoding="utf-8",
            )

            with self.assertRaises(ArtifactSecurityError):
                load_manifested_torch_state(artifact, trusted_roots=[root], map_location="cpu")


if __name__ == "__main__":
    unittest.main()
