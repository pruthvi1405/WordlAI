"""Goal-driven discovery agent loop (Section 3.1): observe -> decide -> act
against a live surface, until the goal is met or a stopping condition fires
(max steps, timeout, or the model calling `escalate`). No hardcoded Wordle
strategy lives here — the model decides what to type each turn from what it
observes.
"""

from __future__ import annotations

import json
import time

from openai import AsyncOpenAI

from wordlehands.agent.tools import TOOL_DEFS, ToolRunner
from wordlehands.config import settings
from wordlehands.evidence.logger import EvidenceLogger
from wordlehands.surface.base import Surface

SYSTEM_PROMPT = """You are operating a live web page for "hello wordl", a Wordle-style word-guessing game, using a small set of tools. You cannot see the page directly — only through the accessibility-tree text and screenshot you are given after each action.

Rules: guess a 5-letter English word. After you submit a guess, each of its 5 letters is shown with a state: "correct" (right letter, right position), "elsewhere" (right letter, wrong position), or "no"/absent (letter not in the word). You have 6 attempts. Use the feedback from every prior guess to choose your next one — track it carefully, this is the actual skill in the game.

How to submit a guess: call type_letters with exactly 5 lowercase letters, then call press_key with "Enter". Then call read_state (the screenshot you receive will also show the result) to see the outcome before deciding your next guess. If a guess is rejected as "not a valid word", it was not a real dictionary word — pick a different real English word next time; it does not use one of your 6 attempts.

When the game has ended (you won, you lost, or you've reached a sensible stopping point), call finish_goal with the outcome. If you become genuinely stuck — repeated unexpected errors, an action you tried was blocked, or you cannot determine what to do next — call escalate instead of guessing randomly forever."""


class DeadEnd(Exception):
    pass


async def run_discovery(
    goal: str,
    target_url: str,
    surface: Surface,
    evidence: EvidenceLogger,
    max_steps: int = 40,
    timeout_s: int = 600,
) -> ToolRunner:
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Put it in a .env file at the project root "
            "(see README.md) before running discovery."
        )

    client = AsyncOpenAI(api_key=settings.openai_api_key)
    tool_runner = ToolRunner(surface, evidence)

    obs = await surface.observe()
    evidence.save_screenshot_b64(obs.screenshot_b64, "step-00-initial.png")
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

    start = time.monotonic()
    step = 0
    while step < max_steps and (time.monotonic() - start) < timeout_s:
        step += 1
        response = await client.chat.completions.create(
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
            break

        obs = await surface.observe()
        evidence.save_screenshot_b64(obs.screenshot_b64, f"step-{step:02d}.png")
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
    else:
        if not tool_runner.finished and not tool_runner.escalated:
            tool_runner.escalated = True
            tool_runner.escalate_reason = (
                f"dead end: reached max_steps={max_steps} or timeout_s={timeout_s} without finishing"
            )
            evidence.log("dead_end", reason=tool_runner.escalate_reason)

    evidence.log(
        "discovery_end",
        finished=tool_runner.finished,
        finish_result=tool_runner.finish_result,
        escalated=tool_runner.escalated,
        escalate_reason=tool_runner.escalate_reason,
        steps_used=step,
    )
    return tool_runner
