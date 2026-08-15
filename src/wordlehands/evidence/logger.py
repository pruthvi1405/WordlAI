"""Evidence / observability (Section 3.5).

Every run (discovery or replay) gets its own timestamped directory under
/evidence with a structured JSONL step log, screenshots, and — on failure —
a richer bundle (accessibility-tree snapshot + DOM excerpt + screenshot) so a
human can debug what the automation actually saw. Everything written here is
passed through `redact()` first (Section 3.4).
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path

from wordlehands.guardrails.redaction import redact, redact_dict
from wordlehands.surface.base import Observation


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


class EvidenceLogger:
    def __init__(self, run_dir: Path):
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "screenshots").mkdir(exist_ok=True)
        self._log_path = self.run_dir / "log.jsonl"
        self._step_counter = 0

    def log(self, event: str, **fields) -> int:
        self._step_counter += 1
        record = {"step": self._step_counter, "event": event, "ts": _now_iso(), **fields}
        record = redact_dict(record)
        with self._log_path.open("a") as f:
            f.write(json.dumps(record, default=str) + "\n")
        return self._step_counter

    def save_screenshot_b64(self, b64: str, name: str) -> Path:
        path = self.run_dir / "screenshots" / name
        path.write_bytes(base64.b64decode(b64))
        return path

    def save_failure_bundle(self, observation: Observation, reason: str) -> None:
        bundle_dir = self.run_dir / "failure"
        bundle_dir.mkdir(exist_ok=True)
        (bundle_dir / "reason.txt").write_text(redact(reason))
        (bundle_dir / "accessibility_snapshot.txt").write_text(
            redact(observation.accessibility_snapshot)
        )
        (bundle_dir / "dom_excerpt.html").write_text(redact(observation.dom_excerpt))
        if observation.screenshot_b64:
            self.save_screenshot_b64(observation.screenshot_b64, "failure.png")

    def write_result(self, result: dict) -> Path:
        result = redact_dict(result)
        path = self.run_dir / "result.json"
        path.write_text(json.dumps(result, indent=2, default=str))
        return path

    def write_json(self, name: str, data: dict) -> Path:
        data = redact_dict(data)
        path = self.run_dir / name
        path.write_text(json.dumps(data, indent=2, default=str))
        return path


def new_run(evidence_root: Path, kind: str) -> EvidenceLogger:
    run_dir = evidence_root / f"{kind}-{_timestamp_slug()}"
    return EvidenceLogger(run_dir)
