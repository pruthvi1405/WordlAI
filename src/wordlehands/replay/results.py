"""The replay result contract (Section 3.3): a discriminated union so callers
can never confuse "the caller needs to know this" (BusinessOutcome) with
"something actually broke" (Failure) — conflating the two is, per the brief's
own glossary, the most common design mistake in this space.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class ReplaySuccess(BaseModel):
    status: Literal["success"] = "success"
    capability_id: str
    version: str
    outputs: dict


class ReplayBusinessOutcome(BaseModel):
    """A real, expected outcome the app itself reported — not a crash. e.g.
    "not a valid word" for a Wordle guess, or "no such member" in the bank
    domain. The caller (agent-facing product) needs this, not an exception."""

    status: Literal["business_outcome"] = "business_outcome"
    capability_id: str
    version: str
    code: str
    detail: str
    outputs: dict = {}


class ReplayFailure(BaseModel):
    """A hard stop: the replay could not confirm it reached the expected
    state, and no known business/recoverable outcome explains why. Carries
    enough detail to debug without re-running: what step, what was expected,
    what was actually observed."""

    status: Literal["failure"] = "failure"
    capability_id: str
    version: str
    step_id: str | None
    expected: str
    observed: str
    message: str


ReplayResult = ReplaySuccess | ReplayBusinessOutcome | ReplayFailure
