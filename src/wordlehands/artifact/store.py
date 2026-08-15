from __future__ import annotations

from pathlib import Path

from wordlehands.artifact.schema import Capability


def save(capability: Capability, artifacts_dir: Path) -> Path:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    path = artifacts_dir / f"{capability.capability_id}.v{capability.version}.json"
    path.write_text(capability.model_dump_json(indent=2))
    return path


def load(path: Path) -> Capability:
    return Capability.model_validate_json(path.read_text())
