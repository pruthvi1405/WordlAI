"""Turns a successful discovery transcript into the versioned `submit_guess`
Capability artifact (Section 3.2).

What's genuinely *derived* from the transcript: proof that the discovery
agent actually used the type-5-letters-then-press-Enter pattern to submit a
guess (`_find_first_submission` below scans the real tool-call trace for it;
`distill_submit_guess_capability` raises if it never happened — we do not
fabricate a capability from a run that didn't demonstrate the interaction).

What's engineering-authored: the checkpoint/output-extraction/error-taxonomy
definitions. This is a deliberate split, documented in REPORT.md
("Artifact schema"): replay must not depend on the LLM to *read* the result
of a guess (that would defeat "no LLM in the decision loop"), so a human/
system author formalizes, once, a deterministic structured read of exactly
the signal the transcript shows the model needed (tile color state, the
alert region) — the same way a real deployment would have an engineer
review and harden a first discovery run into a production capability rather
than trusting the model's screen-reading on every future replay.
"""

from __future__ import annotations

import re

from wordlehands.artifact.schema import (
    Capability,
    Checkpoint,
    CheckpointCondition,
    ErrorTaxonomyEntry,
    ExtractField,
    OutcomeCategory,
    OutputType,
    ParamSpec,
    ParamType,
    Provenance,
    RiskLevel,
    RiskSpec,
    Step,
    StepAction,
    TargetSpec,
)
from wordlehands.surface.base import LocatorSpec, LocatorStrategy


class NoDiscoveredSubmission(Exception):
    pass


def _find_first_submission(calls: list[dict]) -> tuple[int, int]:
    """Scan the real tool-call trace for the first type_letters(5 letters)
    -> press_key('Enter') pair, allowing intervening read_state calls."""
    for i, call in enumerate(calls):
        if call["tool"] != "type_letters" or not call["ok"]:
            continue
        letters = str(call["args"].get("letters", ""))
        if not re.fullmatch(r"[a-z]{5}", letters):
            continue
        for j in range(i + 1, min(i + 4, len(calls))):
            nxt = calls[j]
            if nxt["tool"] == "press_key" and nxt["args"].get("key") == "Enter" and nxt["ok"]:
                return i, j
    raise NoDiscoveredSubmission(
        "no successful type_letters(5 letters) -> press_key('Enter') sequence "
        "found in the discovery transcript; refusing to fabricate a capability"
    )


_ROW_TILES = LocatorSpec(
    strategy=LocatorStrategy.CSS,
    value="table.Game-rows tr.Row:has(td.Row-letter:not(:empty)) >>> td.Row-letter",
    position="last",
    fallbacks=[
        LocatorSpec(
            strategy=LocatorStrategy.CSS,
            value="table.Game-rows tr.Row >>> td.Row-letter",
            position="last",
            robustness_note=(
                "Fallback: last row regardless of content, in case the :not(:empty) "
                "pseudo-class behaves unexpectedly. Less robust than the primary — "
                "kept only as a degrade path."
            ),
        )
    ],
    robustness_note=(
        "Selects the most recently *filled* guess row by content — the row containing "
        "non-empty letter cells — rather than a hardcoded row index. This survives "
        "attempt-number drift entirely: it is correct whether this is guess 1 or guess 6, "
        "and correct whether the prior guess was accepted or rejected (rejected letters "
        "remain visible in the row). The failure mode this avoids is off-by-one row "
        "indexing, the most common way a naive recording breaks on replay."
    ),
)

_ALERT_REGION = LocatorSpec(
    strategy=LocatorStrategy.ROLE,
    value="role=alert",
    position="first",
    fallbacks=[
        LocatorSpec(
            strategy=LocatorStrategy.CSS,
            value="[role='alert']",
            position="first",
            robustness_note="Direct CSS fallback if ARIA role querying fails for any reason.",
        )
    ],
    robustness_note=(
        "The app's own live-region status element — the same node screen readers "
        "announce from. This is a semantic commitment the developers already made for "
        "accessibility; it is far less likely to move than any CSS class name."
    ),
)


def _capability_id() -> str:
    return "submit_guess"


def distill_submit_guess_capability(
    calls: list[dict], discovery_run_id: str, model_used: str, target_url: str
) -> Capability:
    type_idx, press_idx = _find_first_submission(calls)

    steps = [
        Step(
            step_id="type_guess",
            action=StepAction.TYPE_TEXT,
            description=(
                "Type the 5-letter guess via physical key presses — mirrors exactly how "
                f"the discovery agent did it (tool call #{type_idx}); the on-screen keyboard "
                "buttons in this app are aria-hidden/decorative and are not used."
            ),
            input_param="guess",
        ),
        Step(
            step_id="submit",
            action=StepAction.PRESS_KEY,
            description=f"Press Enter to submit the typed guess (tool call #{press_idx} in discovery).",
            literal_key="Enter",
        ),
    ]

    checkpoint = Checkpoint(
        description=(
            "The submitted row's 5 tiles have all been evaluated: every cell's class carries "
            "a 'letter-' state suffix (letter-correct/letter-elsewhere/letter-absent). The base "
            "class 'Row-letter' alone is present on every cell even before evaluation, so "
            "checking for non-emptiness is not sufficient — this checks for the state-specific "
            "substring instead. If this never becomes true, the guess was not accepted "
            "(see error_taxonomy) or something unexpected happened."
        ),
        locator=_ROW_TILES,
        condition=CheckpointCondition.ALL_CONTAIN_SUBSTRING,
        attribute="class",
        match="letter-",
        timeout_ms=4000,
    )

    outputs = [
        ExtractField(
            name="tile_results",
            type=OutputType.ENUM,
            description="Per-letter result for the submitted guess, left to right.",
            locator=_ROW_TILES,
            attribute="class",
            each=True,
            enum_values=["correct", "elsewhere", "absent"],
        ),
        ExtractField(
            name="status_raw",
            type=OutputType.STRING,
            description="Untouched text of the app's status/alert region after this guess, for debuggability.",
            locator=_ALERT_REGION,
            attribute="text",
            each=False,
        ),
        ExtractField(
            name="game_status",
            type=OutputType.ENUM,
            description=(
                "'won' = this guess won the game. 'answer was' = the game ended without "
                "winning (loss) and the answer was revealed. Any other raw value "
                "(status_raw) means the game is still in progress."
            ),
            locator=_ALERT_REGION,
            attribute="text",
            each=False,
            enum_values=["won", "answer was"],
        ),
    ]

    error_taxonomy = [
        ErrorTaxonomyEntry(
            code="invalid_word",
            category=OutcomeCategory.BUSINESS_OUTCOME,
            locator=_ALERT_REGION,
            match_text="Not a valid word",
            description=(
                "The submitted guess is not a recognized dictionary word — a legitimate "
                "result the caller needs (pick a different word), not a crash."
            ),
        ),
        ErrorTaxonomyEntry(
            code="invalid_length",
            category=OutcomeCategory.BUSINESS_OUTCOME,
            locator=_ALERT_REGION,
            match_text="must contain",
            description="The submitted guess was not exactly the required number of letters.",
        ),
        ErrorTaxonomyEntry(
            code="game_already_over",
            category=OutcomeCategory.BUSINESS_OUTCOME,
            locator=_ALERT_REGION,
            match_text="answer was",
            description=(
                "The game had already ended (won, lost, or given up) before this call — "
                "no guess was submitted. Caller should stop invoking this capability."
            ),
        ),
    ]

    return Capability(
        capability_id=_capability_id(),
        name="Submit a Wordle guess and read tile feedback",
        version="1.0.0",
        status="draft",
        description=(
            "Types a 5-letter guess into the current row of a hello-wordl-style game, "
            "submits it, and returns the per-letter correct/elsewhere/absent feedback "
            "plus updated game status. The atomic, reusable interaction underlying the "
            "game — strategy (which word to guess) is the caller's job, not this capability's."
        ),
        target=TargetSpec(
            surface_type="web",
            app_id="hellowordl",
            base_url=target_url,
            allowed_domains=["hellowordl.net"],
        ),
        preconditions=[
            "The target game is loaded and in an 'in_progress' state (not already won/lost/given up).",
            "There is at least one empty guess row remaining.",
        ],
        inputs=[
            ParamSpec(
                name="guess",
                type=ParamType.STRING,
                description="A 5-letter guess word (a-z, case-insensitive — typed as lowercase).",
                pattern="^[A-Za-z]{5}$",
                required=True,
            )
        ],
        steps=steps,
        checkpoint=checkpoint,
        outputs=outputs,
        error_taxonomy=error_taxonomy,
        risk=RiskSpec(
            level=RiskLevel.SAFE,
            requires_confirmation=False,
            rationale=(
                "Submitting a guess is a normal, reversible game action with no destructive "
                "effect — safe for unattended automated replay. Contrast with 'Give up', "
                "which discards game state and is guardrail-blocked (see config/allowlist.yaml)."
            ),
        ),
        provenance=Provenance(
            source="llm_discovery",
            discovery_run_id=discovery_run_id,
            model_used=model_used,
            reviewed_by=None,
        ),
    )
