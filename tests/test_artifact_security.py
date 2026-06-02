import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from bat.analyzer import LSTMStrategy
from bat.research_config import ResearchConfig
from bat.services.run_manager import RunManager


class ArtifactSecurityTest(unittest.TestCase):
    def _write_manifest(self, directory: Path, artifact: Path, sha256: str) -> None:
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "admission": {"passed": True, "reasons": []},
                    "artifacts": {
                        artifact.name: {
                            "sha256": sha256,
                            "runtime_load_allowed": True,
                            "type": "torch_state_dict",
                        }
                    },
                }
            ),
            encoding="utf-8",
        )

    def test_refuses_symlink_artifact(self):
        from bat.services.artifact_security import ArtifactSecurityError, verify_manifested_artifact

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            target = root / "real.pt"
            link = root / "best_model.pt"
            torch.save({"weight": torch.ones(1)}, target)
            link.symlink_to(target)

            with self.assertRaises(ArtifactSecurityError):
                verify_manifested_artifact(link, trusted_roots=[root])

    def test_refuses_artifact_outside_trusted_roots(self):
        from bat.services.artifact_security import ArtifactSecurityError, verify_manifested_artifact

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            trusted = base / "trusted"
            outside = base / "outside"
            trusted.mkdir()
            outside.mkdir()
            artifact = outside / "best_model.pt"
            torch.save({"weight": torch.ones(1)}, artifact)

            with self.assertRaises(ArtifactSecurityError):
                verify_manifested_artifact(artifact, trusted_roots=[trusted])

    def test_refuses_missing_manifest(self):
        from bat.services.artifact_security import ArtifactSecurityError, verify_manifested_artifact

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact = root / "best_model.pt"
            torch.save({"weight": torch.ones(1)}, artifact)

            with self.assertRaises(ArtifactSecurityError):
                verify_manifested_artifact(artifact, trusted_roots=[root])

    def test_refuses_hash_mismatch(self):
        from bat.services.artifact_security import ArtifactSecurityError, verify_manifested_artifact

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact = root / "best_model.pt"
            torch.save({"weight": torch.ones(1)}, artifact)
            self._write_manifest(root, artifact, "0" * 64)

            with self.assertRaises(ArtifactSecurityError):
                verify_manifested_artifact(artifact, trusted_roots=[root])

    def test_load_manifested_torch_state_uses_weights_only(self):
        from bat.services.artifact_security import load_manifested_torch_state, sha256_file

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact = root / "best_model.pt"
            torch.save({"weight": torch.ones(1)}, artifact)
            self._write_manifest(root, artifact, sha256_file(artifact))

            with patch("bat.services.artifact_security.torch.load", return_value={"ok": True}) as load:
                state = load_manifested_torch_state(artifact, trusted_roots=[root], map_location="cpu")

            self.assertEqual(state, {"ok": True})
            self.assertTrue(load.call_args.kwargs["weights_only"])

    def test_load_manifested_torch_state_refuses_non_torch_manifest_type(self):
        from bat.services.artifact_security import ArtifactSecurityError, load_manifested_torch_state, sha256_file

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact = root / "best_model.pt"
            torch.save({"weight": torch.ones(1)}, artifact)
            self._write_manifest(root, artifact, sha256_file(artifact))
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            manifest["artifacts"][artifact.name]["type"] = "pickle"
            (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaises(ArtifactSecurityError):
                load_manifested_torch_state(artifact, trusted_roots=[root], map_location="cpu")

    def test_safe_torch_load_fails_closed_without_weights_only_support(self):
        from bat.services.artifact_security import ArtifactSecurityError, safe_torch_load

        with patch("bat.services.artifact_security.torch.load", side_effect=TypeError("weights_only")):
            with self.assertRaises(ArtifactSecurityError):
                safe_torch_load("model.pt", map_location="cpu")

    def test_runtime_pickle_artifacts_are_refused(self):
        from bat.services.artifact_security import ArtifactSecurityError, verify_manifested_artifact, sha256_file

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            artifact = root / "best_model.pkl"
            artifact.write_bytes(b"pickle payload")
            self._write_manifest(root, artifact, sha256_file(artifact))

            with self.assertRaises(ArtifactSecurityError):
                verify_manifested_artifact(artifact, trusted_roots=[root], runtime=True)

    def test_run_manager_writes_manifest_for_torch_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run = RunManager(root=tmpdir).create_run(
                config=ResearchConfig(),
                model_name="lstm",
                feature_columns=["a", "b"],
                label_distribution={0: 1, 1: 1, 2: 1},
                timestamp="20260102_030405",
            )

            run.save_best_model(torch.nn.Linear(2, 3))

            manifest = json.loads((run.path / "manifest.json").read_text(encoding="utf-8"))
            artifact = manifest["artifacts"]["best_model.pt"]
            self.assertEqual(manifest["feature_columns"], ["a", "b"])
            self.assertTrue(artifact["runtime_load_allowed"])
            self.assertEqual(len(artifact["sha256"]), 64)
            self.assertIn("config_sha256", manifest)
            self.assertIn("training_data_sha256", manifest)
            self.assertIn("git_sha", manifest)

    def test_run_manager_marks_pickle_artifact_unsafe_for_runtime(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run = RunManager(root=tmpdir).create_run(
                config=ResearchConfig(),
                model_name="histgb",
                feature_columns=["a", "b"],
                label_distribution={0: 1, 1: 1, 2: 1},
                timestamp="20260102_030405",
            )

            run.save_sklearn_model({"kind": "histgb"})

            manifest = json.loads((run.path / "manifest.json").read_text(encoding="utf-8"))
            artifact = manifest["artifacts"]["best_model.pkl"]
            self.assertFalse(artifact["runtime_load_allowed"])
            self.assertEqual(artifact["type"], "pickle")

    def test_lstm_strategy_refuses_unmanifested_runtime_model_before_torch_load(self):
        original_cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmpdir:
            os.chdir(tmpdir)
            try:
                data_dir = Path("data")
                data_dir.mkdir()
                torch.save({"weight": torch.ones(1)}, data_dir / "lstm_model.pth")
                strategy = LSTMStrategy(client=None)

                with patch("bat.analyzer.CryptoLSTM") as model_cls, patch(
                    "bat.analyzer.torch.load", side_effect=AssertionError("unsafe torch.load called")
                ):
                    model = strategy._load_model()

                self.assertIsNone(model)
                model_cls.assert_not_called()
            finally:
                os.chdir(original_cwd)


if __name__ == "__main__":
    unittest.main()
