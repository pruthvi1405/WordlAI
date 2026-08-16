"""The Surface abstraction: the seam between "how we perceive/act on a surface"
and everything above it (agent loop, replay executor).

Only PlaywrightWebSurface is implemented in this project. The point of this
module is that nothing above it imports Playwright directly — the agent loop
and the replay executor only ever talk to `Surface`, `Observation`, `Action`,
and `LocatorSpec`. A legacy-web or desktop surface would implement the same
ABC with a different backend and different LocatorStrategy choices; nothing
above this file would need to change. See REPORT.md Section 4.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class LocatorStrategy(str, Enum):
    """How to find/target something on the surface, ordered roughly by
    robustness (most stable first). A LocatorSpec chains a primary strategy
    with fallbacks so replay can degrade gracefully instead of hard-failing
    the moment one strategy stops matching.
    """

    ROLE = "role"  # ARIA role + accessible name — most stable, survives markup/CSS churn
    TEXT = "text"  # visible text content match
    CSS = "css"  # CSS selector — brittle under legacy/no-clean-DOM surfaces
    GLOBAL_KEY = "global_key"  # a physical key event with no element target at all
    COORDINATE = "coordinate"  # last resort: raw x/y — least robust, no semantic anchor


Position = Literal["first", "last"] | int


class LocatorSpec(BaseModel):
    """A single, serializable "how to find/act on this control" description.

    `robustness_note` is required by design — Section 3.2 asks for "reasoning
    about robustness" alongside the locator itself, so it's a first-class
    field, not a comment left out of the artifact.
    """

    strategy: LocatorStrategy
    value: str = Field(
        description=(
            "Meaning depends on strategy: ROLE -> 'role=<aria role>;name=<accessible name>', "
            "TEXT -> substring to match, CSS -> a CSS selector, "
            "GLOBAL_KEY -> the key to send (e.g. 'a', 'Enter', 'Backspace'), "
            "COORDINATE -> 'x,y'."
        )
    )
    position: Position = "first"
    fallbacks: list[LocatorSpec] = Field(default_factory=list)
    robustness_note: str = Field(
        description="Why this locator should survive real-world drift, and what would break it."
    )


LocatorSpec.model_rebuild()


class ActionType(str, Enum):
    PRESS_KEY = "press_key"
    TYPE_TEXT = "type_text"
    CLICK = "click"
    WAIT = "wait"


class Action(BaseModel):
    type: ActionType
    key: str | None = None  # PRESS_KEY
    text: str | None = None  # TYPE_TEXT (sent as a sequence of physical key presses)
    locator: LocatorSpec | None = None  # CLICK
    wait_ms: int | None = None  # WAIT
    reason: str = ""  # why the caller is taking this action (goes into evidence logs)


class Observation(BaseModel):
    url: str
    accessibility_snapshot: str
    dom_excerpt: str
    screenshot_b64: str | None = None
    timestamp: str


class ActionOutcome(BaseModel):
    ok: bool
    message: str
    blocked_by_guardrail: bool = False


class ResolvedField(BaseModel):
    """Result of resolving a LocatorSpec for a read/extraction: the matched
    element(s)' relevant attribute values, in DOM order.
    """

    found: bool
    values: list[str] = Field(default_factory=list)
    matched_strategy: LocatorStrategy | None = None
    note: str = ""


class Surface(ABC):
    """The only interface the agent loop and the replay executor depend on."""

    @abstractmethod
    async def observe(self, include_screenshot: bool = True) -> Observation: ...

    @abstractmethod
    async def act(self, action: Action) -> ActionOutcome: ...

    @abstractmethod
    async def resolve(
        self, locator: LocatorSpec, attribute: str = "text", each: bool = False
    ) -> ResolvedField:
        """Resolve a LocatorSpec (trying fallbacks in order) and read
        `attribute` off the matched element(s) ("text" | "class" | an
        aria-* attribute name). If `each` is True, return one value per
        matched element instead of a single positioned match.
        """
        ...

    @abstractmethod
    async def current_url(self) -> str: ...

    @abstractmethod
    async def screenshot_path(self, path: str) -> None: ...

    @abstractmethod
    async def close(self) -> None: ...
