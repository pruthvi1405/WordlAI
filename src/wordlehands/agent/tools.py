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

import asyncio
import json
import time
from typing import Callable

from wordlehands.agent import solver
from wordlehands.evidence.logger import EvidenceLogger
from wordlehands.surface.base import Action, ActionType, LocatorSpec, LocatorStrategy, Surface

OnCall = Callable[[str, dict, bool, str], None]

# Same CSS locator the distilled submit_guess capability uses for its
# checkpoint/outputs (artifacts/submit_guess.v1.0.0.json) — selects the most
# recently *filled* guess row by content, not a hardcoded row index, so it's
# correct on attempt 1 or attempt 6. Duplicated here (rather than imported
# from the artifact) because this reads tile feedback live during discovery,
# before any artifact exists yet.
_TILE_ROW_LOCATOR = LocatorSpec(
    strategy=LocatorStrategy.CSS,
    value="table.Game-rows tr.Row:has(td.Row-letter:not(:empty)) >>> td.Row-letter",
    position="last",
    robustness_note="Selects the most recently filled row by content, not index.",
)


def _classify_tile_class(raw: str) -> str:
    for state in ("correct", "elsewhere", "absent"):
        if f"letter-{state}" in raw:
            return state
    return "unknown"

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
            "name": "propose_next_words",
            "description": (
                "Get real dictionary words still consistent with every letter-state "
                "constraint learned from guesses submitted so far (correct/elsewhere/absent "
                "positions), ranked by how much they narrow the remaining possibilities. "
                "Call this before choosing your next guess — type_letters will reject any "
                "word that is not a real dictionary word or that contradicts known feedback, "
                "so picking from this list guarantees a submittable guess."
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

    def __init__(self, surface: Surface, evidence: EvidenceLogger, on_call: OnCall | None = None):
        self.surface = surface
        self.evidence = evidence
        self.on_call = on_call
        self.finished = False
        self.finish_result: dict | None = None
        self.escalated = False
        self.escalate_reason: str | None = None
        self.calls: list[dict] = []
        # ReAct solver state (agent/solver.py): every (guess, feedback) pair
        # submitted so far this game, used to both validate the next
        # type_letters call and to answer propose_next_words.
        self._wordlist = solver.load_wordlist()
        self._wordset = set(self._wordlist)
        self._guess_history: list[tuple[str, list[str]]] = []
        self._pending_guess: str | None = None

    async def dispatch(self, name: str, args: dict) -> str:
        if name == "type_letters":
            return await self._do_type(args)
        if name == "press_key":
            return await self._do_press(args)
        if name == "read_state":
            return await self._do_read(args)
        if name == "propose_next_words":
            return await self._do_propose(args)
        if name == "finish_goal":
            self.finished = True
            self.finish_result = args
            self._notify("finish_goal", args, True, args.get("summary", ""))
            return json.dumps({"ok": True})
        if name == "escalate":
            self.escalated = True
            self.escalate_reason = args.get("reason", "")
            self._notify("escalate", args, True, self.escalate_reason)
            return json.dumps({"ok": True})
        return json.dumps({"ok": False, "message": f"unknown tool {name}"})

    async def _do_type(self, args: dict) -> str:
        letters = str(args.get("letters", "")).lower()

        # Guardrail (not the safety allowlist — a correctness gate): reject
        # anything that isn't a real 5-letter dictionary word, or that
        # contradicts a constraint already learned from prior feedback,
        # before it ever reaches the browser. This is what makes "invalid
        # words cannot be entered" literally true rather than just likely.
        if len(letters) != 5 or not letters.isalpha():
            message = f"'{letters}' is not 5 letters — not submitted."
            self._record("type_letters", args, False, message)
            return json.dumps({"ok": False, "message": message})
        if letters not in self._wordset:
            message = f"'{letters}' is not a real dictionary word — not submitted. Call propose_next_words for valid options."
            self._record("type_letters", args, False, message)
            return json.dumps({"ok": False, "message": message})
        if any(
            not solver.consistent_with_guess(letters, g, fb) for g, fb in self._guess_history
        ):
            message = (
                f"'{letters}' contradicts feedback from a previous guess — not submitted. "
                "Call propose_next_words for words consistent with everything learned so far."
            )
            self._record("type_letters", args, False, message)
            return json.dumps({"ok": False, "message": message})

        outcome = await self.surface.act(
            Action(type=ActionType.TYPE_TEXT, text=letters, reason="discovery agent: type guess letters")
        )
        if outcome.ok:
            self._pending_guess = letters
        self._record("type_letters", args, outcome.ok, outcome.message)
        return json.dumps({"ok": outcome.ok, "message": outcome.message})

    async def _do_press(self, args: dict) -> str:
        key = str(args.get("key", ""))
        outcome = await self.surface.act(
            Action(type=ActionType.PRESS_KEY, key=key, reason="discovery agent: press key")
        )
        self._record("press_key", args, outcome.ok, outcome.message)

        if outcome.ok and key == "Enter" and self._pending_guess:
            feedback = await self._read_tile_feedback()
            if feedback:
                self._guess_history.append((self._pending_guess, feedback))
                self.evidence.log(
                    "guess_feedback_captured", guess=self._pending_guess, feedback=feedback
                )
            self._pending_guess = None

        return json.dumps({"ok": outcome.ok, "message": outcome.message})

    async def _read_tile_feedback(self, timeout_ms: int = 2000) -> list[str] | None:
        """Poll the just-submitted row until its tiles are evaluated (same
        checkpoint condition the distilled capability asserts on replay), and
        return per-letter correct/elsewhere/absent states. Returns None if
        the row never evaluates within the timeout (e.g. the game already
        ended) rather than guessing — the solver's history should only ever
        contain real, confirmed feedback.
        """
        deadline = time.monotonic() + timeout_ms / 1000
        while time.monotonic() < deadline:
            resolved = await self.surface.resolve(_TILE_ROW_LOCATOR, attribute="class", each=True)
            if resolved.found and len(resolved.values) == 5:
                states = [_classify_tile_class(v) for v in resolved.values]
                if all(s != "unknown" for s in states):
                    return states
            await asyncio.sleep(0.15)
        return None

    async def _do_read(self, args: dict) -> str:
        obs = await self.surface.observe(include_screenshot=False)
        self.evidence.log("tool_call", tool="read_state", args=args)
        return json.dumps({"accessibility_snapshot": obs.accessibility_snapshot[:3000]})

    async def _do_propose(self, args: dict) -> str:
        result = solver.propose(self._wordlist, self._guess_history)
        self.evidence.log(
            "tool_call",
            tool="propose_next_words",
            args=args,
            ok=True,
            message=f"{result['remaining_possible_count']} candidates remain",
        )
        if self.on_call:
            self.on_call(
                "propose_next_words", args, True, f"{result['remaining_possible_count']} candidates remain"
            )
        return json.dumps(result)

    def _record(self, tool: str, args: dict, ok: bool, message: str) -> None:
        record = {"tool": tool, "args": args, "ok": ok, "message": message}
        self.calls.append(record)
        self._notify(tool, args, ok, message)

    def _notify(self, tool: str, args: dict, ok: bool, message: str) -> None:
        self.evidence.log("tool_call", tool=tool, args=args, ok=ok, message=message)
        if self.on_call:
            self.on_call(tool, args, ok, message)
