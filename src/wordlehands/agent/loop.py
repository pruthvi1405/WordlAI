"""Goal-driven discovery agent loop (Section 3.1): observe -> decide -> act
against a live surface, until the goal is met or a stopping condition fires
(max steps, timeout, or the model calling `escalate`). No hardcoded Wordle
strategy lives here — the model decides what to type each turn from what it
observes.

Escalation handoff (Section 3.6): when the model calls `escalate`, or the
loop hits a dead end, this is where the pause-cede-resume seam actually lives
for discovery. If an `EscalationManager` is supplied, the loop doesn't just
stop — it raises the intervention, blocks until a human resumes control on
the SAME live session, resyncs the solver's guess history from the real
board (the human's actions bypass ToolRunner, so that history would
otherwise go stale), tells the model what happened, and keeps looping. This
is what makes "resume the LLM's own reasoning after a human intervention" a
real path rather than a standalone demo.
"""

from __future__ import annotations

import asyncio
import json
import random
import time
from collections.abc import Callable

from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)

from wordlehands.agent.tools import TOOL_DEFS, OnCall, ToolRunner
from wordlehands.config import settings
from wordlehands.escalation.manager import EscalationManager
from wordlehands.evidence.logger import EvidenceLogger
from wordlehands.surface.base import Surface

SYSTEM_PROMPT = """You are operating a live web page for "hello wordl", a Wordle-style word-guessing game, using a small set of tools. You cannot see the page directly — only through the accessibility-tree text and screenshot you are given after each action.

Rules: guess a 5-letter English word. After you submit a guess, each of its 5 letters is shown with a state: "correct" (right letter, right position), "elsewhere" (right letter, wrong position), or "no"/absent (letter not in the word). You have 6 attempts. Use the feedback from every prior guess to choose your next one — track it carefully, this is the actual skill in the game.

Before choosing a guess, call propose_next_words — it returns real dictionary words that are still consistent with every letter-state constraint learned from your guesses so far, ranked by how much they narrow the remaining possibilities. Picking from that list guarantees your guess will be accepted: type_letters will refuse (without touching the page) any word that isn't a real dictionary word or that contradicts feedback you've already seen.

How to submit a guess: call type_letters with exactly 5 lowercase letters, then call press_key with "Enter". Then call read_state (the screenshot you receive will also show the result) to see the outcome before deciding your next guess.

When the game has ended (you won, you lost, or you've reached a sensible stopping point), call finish_goal with the outcome. If you become genuinely stuck — repeated unexpected errors, an action you tried was blocked, or you cannot determine what to do next — call escalate instead of guessing randomly forever. A human will take over the same session and hand control back to you when they're done; pick up from the state you're shown."""

# Transient failure modes worth a bounded retry: dropped connections, slow
# TLS/proxy hiccups, momentary rate limits, and the provider's own 5xx —
# none of these mean the request was bad, just that it didn't land. Anything
# else (auth errors, malformed requests) should fail immediately rather than
# retry into the same wall.
_RETRYABLE_ERRORS = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)


class DeadEnd(Exception):
    pass


def _save_screenshot_if_present(evidence: EvidenceLogger, obs, name: str) -> None:
    # PlaywrightWebSurface always returns a screenshot for observe()'s
    # default include_screenshot=True, but Observation.screenshot_b64 is
    # typed Optional (a Surface implementation is free to omit it) — guard
    # rather than assume, same pattern as EvidenceLogger.save_failure_bundle.
    if obs.screenshot_b64:
        evidence.save_screenshot_b64(obs.screenshot_b64, name)


async def _create_completion_with_retry(
    client: AsyncOpenAI, evidence: EvidenceLogger, *, max_attempts: int = 4, **kwargs
):
    for attempt in range(1, max_attempts + 1):
        try:
            return await client.chat.completions.create(**kwargs)
        except _RETRYABLE_ERRORS as exc:
            if attempt == max_attempts:
                raise
            delay = min(2**attempt, 15) + random.uniform(0, 1)
            evidence.log(
                "llm_call_retry",
                attempt=attempt,
                max_attempts=max_attempts,
                error_type=type(exc).__name__,
                error=str(exc)[:300],
                delay_s=round(delay, 1),
            )
            await asyncio.sleep(delay)


async def run_discovery(
    goal: str,
    target_url: str,
    surface: Surface,
    evidence: EvidenceLogger,
    max_steps: int = 40,
    timeout_s: int = 600,
    on_call: OnCall | None = None,
    escalation_manager: EscalationManager | None = None,
    on_escalate: Callable[[str, str], None] | None = None,
    operator_url: str | None = None,
) -> ToolRunner:
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Put it in a .env file at the project root "
            "(see README.md) before running discovery."
        )

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    tool_runner = ToolRunner(surface, evidence, on_call=on_call)

    obs = await surface.observe()
    _save_screenshot_if_present(evidence, obs, "step-00-initial.png")
    evidence.log("discovery_start", goal=goal, target_url=target_url, model=settings.openai_model)

    messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"Goal: {goal}\n\nCurrent accessibility snapshot:\n{obs.accessibility_snapshot}",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{obs.screenshot_b64}"},
                },
            ],
        },
    ]

    # Both escalation triggers — the model explicitly calling `escalate`
    # mid-turn, and running out of step/time budget without finishing — are
    # handled through the same top-of-loop check, so a human hand-off works
    # identically either way and a resume can grant a fresh budget rather
    # than the loop just re-hitting the same wall next iteration.
    start = time.monotonic()
    step = 0
    budget_steps = max_steps
    while True:
        if tool_runner.finished:
            break

        out_of_budget = step >= budget_steps or (time.monotonic() - start) >= timeout_s
        if out_of_budget and not tool_runner.escalated:
            tool_runner.escalated = True
            tool_runner.escalate_reason = (
                f"dead end: reached max_steps={budget_steps} or timeout_s={timeout_s} without finishing"
            )
            evidence.log("dead_end", reason=tool_runner.escalate_reason)

        if tool_runner.escalated:
            if escalation_manager is None:
                break
            await _hand_off_to_human_and_resume(
                surface, evidence, tool_runner, escalation_manager, messages, on_escalate, operator_url
            )
            # A human just intervened — give the model a fresh runway
            # instead of immediately re-hitting the same step/time wall.
            budget_steps = step + max_steps
            start = time.monotonic()
            continue

        step += 1
        response = await _create_completion_with_retry(
            client,
            evidence,
            model=settings.openai_model,
            messages=messages,
            tools=TOOL_DEFS,
            tool_choice="auto",
        )
        msg = response.choices[0].message
        evidence.log(
            "llm_turn",
            step=step,
            content=msg.content,
            tool_calls=[tc.function.name for tc in (msg.tool_calls or [])],
        )
        messages.append(msg.model_dump(exclude_none=True))

        if not msg.tool_calls:
            messages.append(
                {
                    "role": "user",
                    "content": "Please call one of the available tools to continue "
                    "(type_letters, press_key, read_state, finish_goal, or escalate).",
                }
            )
            continue

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments or "{}")
            result = await tool_runner.dispatch(tc.function.name, args)
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})

        if tool_runner.finished or tool_runner.escalated:
            continue  # handled at the top of the next iteration

        obs = await surface.observe()
        _save_screenshot_if_present(evidence, obs, f"step-{step:02d}.png")
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"Accessibility snapshot:\n{obs.accessibility_snapshot}"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{obs.screenshot_b64}"},
                    },
                ],
            }
        )

    evidence.log(
        "discovery_end",
        finished=tool_runner.finished,
        finish_result=tool_runner.finish_result,
        escalated=tool_runner.escalated,
        escalate_reason=tool_runner.escalate_reason,
        steps_used=step,
    )
    return tool_runner


async def _hand_off_to_human_and_resume(
    surface: Surface,
    evidence: EvidenceLogger,
    tool_runner: ToolRunner,
    escalation_manager: EscalationManager,
    messages: list[dict],
    on_escalate: Callable[[str, str], None] | None,
    operator_url: str | None,
) -> None:
    reason = tool_runner.escalate_reason or "discovery agent escalated"
    await escalation_manager.request_intervention(reason=reason, capability_id="discovery")
    if on_escalate:
        on_escalate(reason, operator_url or "")

    evidence.log("waiting_for_human_resume")
    await escalation_manager.wait_for_resume()
    evidence.log("human_resumed", human_action_count=len(escalation_manager.human_actions))

    # The human acted directly on the raw surface via EscalationManager/the
    # operator server — ToolRunner never saw those actions, so its
    # incrementally-tracked guess history is now stale. Re-derive it from
    # the actual board rather than the model's own tool-call trace.
    await tool_runner.resync_guess_history_from_board()

    tool_runner.escalated = False
    tool_runner.escalate_reason = None

    obs = await surface.observe()
    if escalation_manager.human_actions:
        human_summary = "; ".join(
            f"{a['description']} -> {'ok' if a['ok'] else 'failed'}" for a in escalation_manager.human_actions
        )
    else:
        human_summary = "no actions were recorded"
    messages.append(
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        "A human operator took control of this session, performed the "
                        f"following action(s), then handed control back to you: {human_summary}. "
                        "Your known guess/feedback history has been resynced from the live board, "
                        "so propose_next_words already reflects anything the human did. Continue "
                        "toward the original goal from the current state below.\n\n"
                        f"Accessibility snapshot:\n{obs.accessibility_snapshot}"
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{obs.screenshot_b64}"},
                },
            ],
        }
    )
