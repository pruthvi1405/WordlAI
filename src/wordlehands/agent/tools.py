"""The tool palette exposed to the discovery LLM (Section 3.1).

Design note (see REPORT.md "Architecture"): tools are deliberately low-level
and locator-free where possible — `type_letters`/`press_key` send physical
key events, mirroring exactly how a human plays this game (the on-screen
keyboard in the target app is `aria-hidden`/decorative; real input is
physical keydown, which we discovered by probing the live DOM before writing
this). There is no generic `click(x, y)` or raw-coordinate tool: the one
click-worthy action in this app ("Give up") is destructive/irreversible, and
by simply not handing the model a click tool we get a structural guarantee
against it, on top of (not instead of) the allowlist guardrail.

The model still does the real discovery work: deciding *what* to type each
turn (strategy — the genuinely dynamic part of Wordle), *when* to submit,
and *how to interpret* the screenshot/accessibility-tree feedback it gets
back. `agent/distiller.py` is what turns a successful transcript into the
deterministic, replayable `submit_guess` Capability.
"""

from __future__ import annotations

import json

from wordlehands.evidence.logger import EvidenceLogger
from wordlehands.surface.base import Action, ActionType, Surface

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "type_letters",
            "description": (
                "Type letters on the physical keyboard, one keydown per character, "
                "into whatever currently has keyboard focus on the page. Lowercase a-z only."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "letters": {
                        "type": "string",
                        "description": "Lowercase letters to type, e.g. 'crane'.",
                    }
                },
                "required": ["letters"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "press_key",
            "description": "Press a single named key: 'Enter' to submit a guess, 'Backspace' to delete the last typed letter.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string", "enum": ["Enter", "Backspace"]}},
                "required": ["key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_state",
            "description": (
                "Re-read the current page state as accessibility-tree text. "
                "A fresh screenshot is also attached to your next message automatically after every tool call."
            ),
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish_goal",
            "description": "Call once the goal is achieved or a reasonable stopping point is reached (e.g. the game ended). Ends the run.",
            "parameters": {
                "type": "object",
                "properties": {
                    "outcome": {"type": "string", "enum": ["won", "lost", "reasonable_stop"]},
                    "summary": {"type": "string"},
                },
                "required": ["outcome", "summary"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate",
            "description": (
                "Call if you are stuck and cannot safely make progress on your own "
                "(repeated unexpected errors, an action was blocked by policy, or you "
                "genuinely don't know what to do next). A human will be asked to take over."
            ),
            "parameters": {
                "type": "object",
                "properties": {"reason": {"type": "string"}},
                "required": ["reason"],
            },
        },
    },
]


class ToolRunner:
    """Dispatches model tool calls onto a (guardrail-wrapped) Surface, and
    keeps the trace the distiller later turns into a Capability artifact.
    """

    def __init__(self, surface: Surface, evidence: EvidenceLogger):
        self.surface = surface
        self.evidence = evidence
        self.finished = False
        self.finish_result: dict | None = None
        self.escalated = False
        self.escalate_reason: str | None = None
        self.calls: list[dict] = []

    async def dispatch(self, name: str, args: dict) -> str:
        if name == "type_letters":
            return await self._do_type(args)
        if name == "press_key":
            return await self._do_press(args)
        if name == "read_state":
            return await self._do_read(args)
        if name == "finish_goal":
            self.finished = True
            self.finish_result = args
            self.evidence.log("tool_call", tool=name, args=args)
            return json.dumps({"ok": True})
        if name == "escalate":
            self.escalated = True
            self.escalate_reason = args.get("reason", "")
            self.evidence.log("tool_call", tool=name, args=args)
            return json.dumps({"ok": True})
        return json.dumps({"ok": False, "message": f"unknown tool {name}"})

    async def _do_type(self, args: dict) -> str:
        letters = str(args.get("letters", "")).lower()
        outcome = await self.surface.act(
            Action(type=ActionType.TYPE_TEXT, text=letters, reason="discovery agent: type guess letters")
        )
        self._record("type_letters", args, outcome.ok, outcome.message)
        return json.dumps({"ok": outcome.ok, "message": outcome.message})

    async def _do_press(self, args: dict) -> str:
        key = str(args.get("key", ""))
        outcome = await self.surface.act(
            Action(type=ActionType.PRESS_KEY, key=key, reason="discovery agent: press key")
        )
        self._record("press_key", args, outcome.ok, outcome.message)
        return json.dumps({"ok": outcome.ok, "message": outcome.message})

    async def _do_read(self, args: dict) -> str:
        obs = await self.surface.observe(include_screenshot=False)
        self.evidence.log("tool_call", tool="read_state", args=args)
        return json.dumps({"accessibility_snapshot": obs.accessibility_snapshot[:3000]})

    def _record(self, tool: str, args: dict, ok: bool, message: str) -> None:
        record = {"tool": tool, "args": args, "ok": ok, "message": message}
        self.calls.append(record)
        self.evidence.log("tool_call", **record)
