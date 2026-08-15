"""Human-in-the-loop escalation & handoff (Section 3.6).

The control-transfer model is an explicit state machine on a single shared
live session: `automation -> (escalate) -> human -> (resume) -> automation`.
Nothing about the Playwright page/context changes on escalation — the same
browser tab stays open, and a human (via `operator_server.py`'s tiny FastAPI
passthrough, or by touching the visible headed browser window directly) acts
on that exact session. Every human action taken while `control_owner ==
"human"` is logged through the same evidence pipeline as automated steps, so
"what did the human do" is part of the run's record, not a gap in it.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from wordlehands.evidence.logger import EvidenceLogger
from wordlehands.surface.base import Surface


class EscalationManager:
    def __init__(self, surface: Surface, evidence: EvidenceLogger):
        self.surface = surface  # the raw, unguarded surface — a human in control is accountable directly
        self.evidence = evidence
        self.control_owner = "automation"
        self.reason: str | None = None
        self.human_actions: list[dict] = []
        self._resume_event = asyncio.Event()

    async def request_intervention(
        self, reason: str, capability_id: str | None = None, step_id: str | None = None
    ) -> dict:
        obs = await self.surface.observe()
        self.reason = reason
        self.control_owner = "human"
        self._resume_event.clear()
        request = {
            "reason": reason,
            "capability_id": capability_id,
            "step_id": step_id,
            "url": obs.url,
            "requested_at": datetime.now(timezone.utc).isoformat(),
        }
        self.evidence.write_json("intervention_request.json", request)
        if obs.screenshot_b64:
            self.evidence.save_screenshot_b64(obs.screenshot_b64, "intervention.png")
        (self.evidence.run_dir / "intervention_ax_snapshot.txt").write_text(obs.accessibility_snapshot)
        self.evidence.log("escalation_requested", **request)
        return request

    async def record_human_action(self, description: str, ok: bool, message: str) -> None:
        record = {
            "description": description,
            "ok": ok,
            "message": message,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        self.human_actions.append(record)
        self.evidence.log("human_action", **record)

    async def resume(self) -> None:
        self.control_owner = "automation"
        self.evidence.log("control_resumed_to_automation", human_action_count=len(self.human_actions))
        self.evidence.write_json("human_actions.json", {"actions": self.human_actions})
        self._resume_event.set()

    async def wait_for_resume(self) -> None:
        await self._resume_event.wait()
