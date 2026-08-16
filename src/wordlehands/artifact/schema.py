"""The Capability artifact schema (Section 3.2) — the focal point of this
project's design. A Capability is not "a recording of one run"; it's a typed,
versioned, agent-invocable contract: given these inputs, against this target,
following these steps, we assert this checkpoint, and return these outputs —
or one of these known error/business outcomes.

Granularity decision (see REPORT.md "Artifact schema"): a Capability here is
the atomic UI interaction ("submit one guess, read back the tile feedback"),
not "solve the whole puzzle" — the puzzle's answer changes per session, so a
recorded winning sequence isn't reusable, but the interaction mechanics are.
This mirrors the brief's own bank example ("look up member, read balance"):
deterministic mechanics, real data back.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field

from wordlehands.surface.base import LocatorSpec


class StepAction(str, Enum):
    PRESS_KEY = "press_key"
    TYPE_TEXT = "type_text"
    CLICK = "click"
    WAIT = "wait"


class Step(BaseModel):
    step_id: str
    action: StepAction
    description: str
    locator: LocatorSpec | None = Field(
        default=None, description="Required for CLICK. Unused otherwise."
    )
    input_param: str | None = Field(
        default=None,
        description="For TYPE_TEXT: name of the input param whose value is typed, "
        "character by character, as physical key presses.",
    )
    literal_key: str | None = Field(
        default=None, description="For PRESS_KEY: a fixed key, e.g. 'Enter'."
    )
    wait_ms: int | None = Field(default=None, description="For WAIT.")


class CheckpointCondition(str, Enum):
    ALL_MATCH_NONEMPTY_ATTR = "all_match_nonempty_attr"  # every matched element has a non-empty `attribute`
    ALL_CONTAIN_SUBSTRING = "all_contain_substring"  # every matched element's `attribute` contains `match`
    TEXT_PRESENT = "text_present"  # resolved text equals/contains `match`


class Checkpoint(BaseModel):
    """Asserts the run actually reached the expected state — never assume a
    click/keypress worked just because it didn't raise."""

    description: str
    locator: LocatorSpec
    condition: CheckpointCondition
    attribute: str = "class"
    match: str | None = None
    timeout_ms: int = 5000


class OutcomeCategory(str, Enum):
    BUSINESS_OUTCOME = "business_outcome"  # expected, legitimate result the caller needs
    RECOVERABLE = "recoverable"  # transient/known interstitial; replay handles it, doesn't surface as error
    HARD_FAILURE = "hard_failure"  # stop and surface a clear, debuggable error


class ErrorTaxonomyEntry(BaseModel):
    code: str
    category: OutcomeCategory
    locator: LocatorSpec
    match_text: str
    description: str


class ParamType(str, Enum):
    STRING = "string"
    INTEGER = "integer"
    BOOLEAN = "boolean"


class ParamSpec(BaseModel):
    name: str
    type: ParamType
    description: str
    pattern: str | None = Field(default=None, description="Optional regex validation.")
    required: bool = True


class OutputType(str, Enum):
    STRING = "string"
    ENUM = "enum"
    STRING_LIST = "string_list"


class ExtractField(BaseModel):
    """Both the typed output *declaration* (name/type/shape) and the
    *instructions* for how the replay engine extracts it — kept together so
    the artifact stays self-describing rather than needing a second schema
    doc that can drift from the extraction code."""

    name: str
    type: OutputType
    description: str
    locator: LocatorSpec
    attribute: str = "text"
    each: bool = False
    enum_values: list[str] | None = None


class RiskLevel(str, Enum):
    SAFE = "safe"
    REVERSIBLE = "reversible"
    RISKY = "risky"
    IRREVERSIBLE = "irreversible"


class RiskSpec(BaseModel):
    level: RiskLevel
    requires_confirmation: bool
    rationale: str


class TargetSpec(BaseModel):
    surface_type: Literal["web", "legacy_web", "desktop"] = "web"
    app_id: str
    base_url: str
    allowed_domains: list[str]


class Provenance(BaseModel):
    created_at: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    source: Literal["llm_discovery", "hand_authored"]
    discovery_run_id: str | None = None
    model_used: str | None = None
    reviewed_by: str | None = None


class Capability(BaseModel):
    capability_id: str
    name: str
    version: str = Field(description="Semver, e.g. '1.0.0'.")
    status: Literal["draft", "approved"] = "draft"
    description: str

    target: TargetSpec
    preconditions: list[str] = Field(default_factory=list)

    inputs: list[ParamSpec]
    steps: list[Step]
    checkpoint: Checkpoint
    outputs: list[ExtractField]
    error_taxonomy: list[ErrorTaxonomyEntry] = Field(default_factory=list)

    risk: RiskSpec
    provenance: Provenance

    def input_names(self) -> set[str]:
        return {p.name for p in self.inputs}
