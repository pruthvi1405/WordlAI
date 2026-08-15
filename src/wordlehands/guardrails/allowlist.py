"""Safety & policy guardrails (Section 3.4).

`GuardedSurface` wraps a real `Surface` and enforces policy on every single
`act()` call before it reaches the real surface — the agent loop and the
replay executor are both routed through this wrapper, so there is exactly one
enforcement point rather than scattered checks.

Policy: an explicit domain allowlist, an explicit action-type allowlist, and a
risk classifier that separates safe/reversible actions (typing a guess,
pressing keys, waiting) from risky/irreversible ones (anything matching
`risky_click_names`, e.g. "Give up" — which discards in-progress game state
and stands in for an irreversible action in a real banking flow, like
"Submit disbursement"). Risky actions are always blocked from unattended
execution in this project (never auto-confirmed) — the conservative choice
the brief explicitly allows ("block, require confirmation, or flag — your
call, justify it"). Blocking is chosen over auto-confirming or silently
flagging-and-proceeding because an irreversible action in a regulated banking
context should never be a default outcome of an LLM's judgment call.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from wordlehands.surface.base import (
    Action,
    ActionOutcome,
    ActionType,
    LocatorStrategy,
    Observation,
    ResolvedField,
    Surface,
)


class GuardrailViolation(Exception):
    pass


class AllowlistPolicy(BaseModel):
    allowed_domains: list[str]
    allowed_action_types: list[str]
    safe_action_types: list[str]
    risky_click_names: list[str]
    denied_domains: list[str]

    @classmethod
    def load(cls, path: Path) -> "AllowlistPolicy":
        data = yaml.safe_load(path.read_text())
        return cls(**data)

    def is_domain_allowed(self, url: str) -> bool:
        if any(d == "*" for d in self.denied_domains) and not any(
            d in url for d in self.allowed_domains
        ):
            return False
        return True

    def is_action_type_allowed(self, action_type: ActionType) -> bool:
        return action_type.value in self.allowed_action_types

    def is_risky_click(self, locator_value: str) -> bool:
        # locator_value is the raw LocatorSpec.value, e.g. "role=button;name=Give up"
        return any(name.lower() in locator_value.lower() for name in self.risky_click_names)


class GuardedSurface(Surface):
    """Decorator over a real Surface that enforces AllowlistPolicy on every act()."""

    def __init__(self, inner: Surface, policy: AllowlistPolicy, on_violation=None):
        self._inner = inner
        self._policy = policy
        self._on_violation = on_violation  # optional callback(action, reason) for logging

    async def observe(self, include_screenshot: bool = True) -> Observation:
        return await self._inner.observe(include_screenshot=include_screenshot)

    async def resolve(self, locator, attribute: str = "text", each: bool = False) -> ResolvedField:
        return await self._inner.resolve(locator, attribute=attribute, each=each)

    async def current_url(self) -> str:
        return await self._inner.current_url()

    async def screenshot_path(self, path: str) -> None:
        await self._inner.screenshot_path(path)

    async def close(self) -> None:
        await self._inner.close()

    async def act(self, action: Action) -> ActionOutcome:
        url = await self._inner.current_url()
        if not self._policy.is_domain_allowed(url):
            return self._block(action, f"domain not allowlisted: {url}")

        if not self._policy.is_action_type_allowed(action.type):
            return self._block(action, f"action type not allowlisted: {action.type.value}")

        if action.type == ActionType.CLICK and action.locator is not None:
            all_specs = [action.locator, *action.locator.fallbacks]
            if any(
                s.strategy == LocatorStrategy.ROLE and self._policy.is_risky_click(s.value)
                for s in all_specs
            ):
                return self._block(
                    action,
                    "click target classified risky/irreversible — blocked, requires human escalation",
                )

        return await self._inner.act(action)

    def _block(self, action: Action, reason: str) -> ActionOutcome:
        if self._on_violation:
            self._on_violation(action, reason)
        return ActionOutcome(ok=False, message=reason, blocked_by_guardrail=True)
