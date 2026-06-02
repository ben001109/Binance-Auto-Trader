from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import torch


class ArtifactSecurityError(ValueError):
    pass


@dataclass(frozen=True)
class VerifiedArtifact:
    path: Path
    manifest: dict
    artifact: dict


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def safe_torch_load(path: str | Path, *, map_location=None):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError as exc:
        raise ArtifactSecurityError("torch.load weights_only=True is not supported") from exc


def verify_manifested_artifact(
    artifact_path: str | Path,
    *,
    trusted_roots: list[str | Path] | tuple[str | Path, ...] | None = None,
    runtime: bool = True,
) -> VerifiedArtifact:
    path = Path(artifact_path)
    if path.is_symlink():
        raise ArtifactSecurityError(f"Artifact path uses symlink: {path}")
    if not path.exists():
        raise ArtifactSecurityError(f"Artifact does not exist: {path}")

    resolved = path.resolve(strict=True)
    roots = [Path(root).resolve() for root in (trusted_roots or ("data", "runs"))]
    if not any(_is_relative_to(resolved, root) for root in roots):
        raise ArtifactSecurityError(f"Artifact outside trusted roots: {path}")

    manifest_path = resolved.parent / "manifest.json"
    if manifest_path.is_symlink():
        raise ArtifactSecurityError(f"Manifest path uses symlink: {manifest_path}")
    if not manifest_path.exists():
        raise ArtifactSecurityError(f"Artifact manifest missing: {manifest_path}")

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ArtifactSecurityError(f"Artifact manifest is invalid JSON: {manifest_path}") from exc

    artifact = manifest.get("artifacts", {}).get(resolved.name)
    if not isinstance(artifact, dict):
        raise ArtifactSecurityError(f"Artifact missing from manifest: {resolved.name}")

    expected_hash = artifact.get("sha256")
    if not expected_hash or sha256_file(resolved) != expected_hash:
        raise ArtifactSecurityError(f"Artifact hash mismatch: {resolved.name}")

    if runtime:
        if resolved.suffix == ".pkl" or not artifact.get("runtime_load_allowed", False):
            raise ArtifactSecurityError(f"Artifact is not allowed for runtime loading: {resolved.name}")
        admission = manifest.get("admission", {})
        if not isinstance(admission, dict) or not admission.get("passed", False):
            raise ArtifactSecurityError(f"Artifact has not passed model admission: {resolved.name}")

    return VerifiedArtifact(path=resolved, manifest=manifest, artifact=artifact)


def load_manifested_torch_state(
    artifact_path: str | Path,
    *,
    trusted_roots: list[str | Path] | tuple[str | Path, ...] | None = None,
    map_location=None,
):
    verified = verify_manifested_artifact(
        artifact_path,
        trusted_roots=trusted_roots,
        runtime=True,
    )
    if verified.path.suffix not in {".pt", ".pth"}:
        raise ArtifactSecurityError(f"Artifact is not a torch state file: {verified.path.name}")
    if verified.artifact.get("type") not in {"torch_state_dict", "torch_checkpoint"}:
        raise ArtifactSecurityError(f"Artifact manifest type is not torch-safe: {verified.path.name}")
    return safe_torch_load(verified.path, map_location=map_location)


def write_artifact_manifest(
    directory: str | Path,
    artifacts: list[dict],
    *,
    feature_columns: list[str] | None = None,
    config_sha256: str | None = None,
    training_data_sha256: str | None = None,
    git_sha: str | None = None,
    admission: dict | None = None,
) -> dict:
    directory = Path(directory)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "artifacts": {},
        }

    manifest["feature_columns"] = list(feature_columns or manifest.get("feature_columns", []))
    manifest["config_sha256"] = config_sha256 if config_sha256 is not None else manifest.get("config_sha256")
    manifest["training_data_sha256"] = training_data_sha256
    manifest["git_sha"] = git_sha if git_sha is not None else manifest.get("git_sha")
    if admission is not None:
        manifest["admission"] = admission

    artifact_map = manifest.setdefault("artifacts", {})
    for artifact in artifacts:
        path = Path(artifact["path"])
        artifact_map[path.name] = {
            "sha256": sha256_file(path),
            "type": artifact.get("type", "unknown"),
            "runtime_load_allowed": bool(artifact.get("runtime_load_allowed", False)),
        }

    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return manifest


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False
