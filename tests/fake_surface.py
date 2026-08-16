"""A scriptable in-memory Surface for fast, browser-free tests of the replay
executor and guardrails. Mirrors just enough of hellowordl.net's real
locators (Row-letter tiles, the alert region) for the submit_guess
capability's checkpoint/output/error_taxonomy logic to be exercised exactly
as it would be live.
"""

from __future__ import annotations

from wordlehands.surface.base import (
    Action,
    ActionOutcome,
    LocatorStrategy,
    Observation,
    ResolvedField,
    Surface,
)

# A real (tiny, 1x1) PNG rather than None — run_discovery (agent/loop.py)
# writes every observation's screenshot to evidence unconditionally, same as
# the real PlaywrightWebSurface always provides one.
_TINY_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII="

ALERT_TEXT = {
    "success": "",
    "invalid_word": "Not a valid word",
    "game_already_over": "You won! The answer was ABYSS. (Enter to play again)",
    "hard_failure": "",
}


class FakeSurface(Surface):
    def __init__(self, mode: str = "success", url: str = "https://hellowordl.net/"):
        self.mode = mode
        self.url = url
        self.actions: list[Action] = []
        self.tile_classes = [
            "Row-letter letter-correct",
            "Row-letter letter-elsewhere",
            "Row-letter letter-absent",
            "Row-letter letter-absent",
            "Row-letter letter-correct",
        ]

    async def observe(self, include_screenshot: bool = True) -> Observation:
        return Observation(
            url=self.url,
            accessibility_snapshot="fake ax snapshot",
            dom_excerpt="<div>fake</div>",
            screenshot_b64=_TINY_PNG_B64 if include_screenshot else None,
            timestamp="2026-01-01T00:00:00Z",
        )

    async def act(self, action: Action) -> ActionOutcome:
        self.actions.append(action)
        return ActionOutcome(ok=True, message="ok")

    async def resolve(self, locator, attribute: str = "text", each: bool = False) -> ResolvedField:
        if locator.strategy == LocatorStrategy.CSS and "Row-letter" in locator.value:
            if self.mode == "hard_failure":
                return ResolvedField(found=False, note="simulated: element never appears")
            if attribute == "class":
                if self.mode == "success":
                    return ResolvedField(found=True, values=list(self.tile_classes))
                return ResolvedField(found=True, values=["Row-letter"] * 5)
            if attribute == "text":
                return ResolvedField(found=True, values=list("crane"))

        if locator.strategy == LocatorStrategy.ROLE and "alert" in locator.value:
            return ResolvedField(found=True, values=[ALERT_TEXT[self.mode]])

        return ResolvedField(found=False, note="unhandled by FakeSurface")

    async def current_url(self) -> str:
        return self.url

    async def screenshot_path(self, path: str) -> None:
        pass

    async def close(self) -> None:
        pass
