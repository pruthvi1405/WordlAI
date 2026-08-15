from wordlehands.config import settings
from wordlehands.guardrails.allowlist import AllowlistPolicy, GuardedSurface
from wordlehands.surface.base import Action, ActionType, LocatorSpec, LocatorStrategy

from .fake_surface import FakeSurface


def load_policy():
    return AllowlistPolicy.load(settings.allowlist_path)


async def test_allowed_action_passes_through():
    surface = FakeSurface(url="https://hellowordl.net/")
    guarded = GuardedSurface(surface, load_policy())

    outcome = await guarded.act(Action(type=ActionType.TYPE_TEXT, text="crane"))

    assert outcome.ok
    assert len(surface.actions) == 1


async def test_action_blocked_when_domain_not_allowlisted():
    surface = FakeSurface(url="https://evil.example.com/")
    guarded = GuardedSurface(surface, load_policy())

    outcome = await guarded.act(Action(type=ActionType.PRESS_KEY, key="Enter"))

    assert not outcome.ok
    assert outcome.blocked_by_guardrail
    assert surface.actions == []


async def test_risky_click_is_blocked_never_executed():
    surface = FakeSurface(url="https://hellowordl.net/")
    guarded = GuardedSurface(surface, load_policy())

    give_up = LocatorSpec(
        strategy=LocatorStrategy.ROLE,
        value="role=button;name=Give up",
        robustness_note="test",
    )
    outcome = await guarded.act(Action(type=ActionType.CLICK, locator=give_up))

    assert not outcome.ok
    assert outcome.blocked_by_guardrail
    assert surface.actions == []  # the real surface never even saw the risky click


async def test_violation_callback_is_invoked():
    surface = FakeSurface(url="https://hellowordl.net/")
    violations = []
    guarded = GuardedSurface(
        surface, load_policy(), on_violation=lambda action, reason: violations.append(reason)
    )

    give_up = LocatorSpec(strategy=LocatorStrategy.ROLE, value="role=button;name=Give up", robustness_note="test")
    await guarded.act(Action(type=ActionType.CLICK, locator=give_up))

    assert len(violations) == 1
    assert "risky" in violations[0]
