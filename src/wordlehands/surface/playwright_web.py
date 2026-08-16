"""The only concrete Surface implementation in this project: a Playwright-backed
web surface. Everything Playwright-specific lives here — nothing above this
module (agent loop, replay executor) imports playwright directly.

Locator encoding conventions (documented here because LocatorSpec.value is a
plain string, not a structured object — kept that way so the schema stays
simple and JSON-serializable):

  ROLE   "role=<aria role>;name=<accessible name>"   e.g. "role=button;name=Give up"
  TEXT   "<substring>"                                 matched via get_by_text
  CSS    "<selector>"  or  "<outer selector> >>> <inner selector>"
         The ">>> " form resolves `position` against the OUTER selector first
         (e.g. "the last row that has any letters in it" — a content-relative,
         state-relative locator that survives attempt-count/row-index drift),
         then reads the INNER selector's matches underneath it.
  GLOBAL_KEY  "<key>"   used only for act(PRESS_KEY); not resolvable for reads.
  COORDINATE  "<x>,<y>" last-resort click target; not resolvable for reads.
"""

from __future__ import annotations

import base64
from datetime import datetime, timezone

from playwright.async_api import Locator, Page

from wordlehands.surface.base import (
    Action,
    ActionOutcome,
    ActionType,
    LocatorSpec,
    LocatorStrategy,
    Observation,
    ResolvedField,
    Surface,
)

DOM_EXCERPT_MAX_CHARS = 6000


def _parse_role_value(value: str) -> tuple[str, str | None]:
    role = value
    name = None
    for part in value.split(";"):
        part = part.strip()
        if part.startswith("role="):
            role = part[len("role=") :]
        elif part.startswith("name="):
            name = part[len("name=") :]
    return role, name


def _apply_position(locator: Locator, position) -> Locator:
    if position == "first":
        return locator.first
    if position == "last":
        return locator.last
    return locator.nth(int(position))


class PlaywrightWebSurface(Surface):
    def __init__(self, page: Page):
        self._page = page

    @property
    def page(self) -> Page:
        return self._page

    async def observe(self, include_screenshot: bool = True) -> Observation:
        ax = await self._page.locator("body").aria_snapshot()
        try:
            dom = await self._page.locator(".Game").inner_html()
        except Exception:
            dom = await self._page.content()
        dom = dom[:DOM_EXCERPT_MAX_CHARS]

        screenshot_b64 = None
        if include_screenshot:
            png = await self._page.screenshot()
            screenshot_b64 = base64.b64encode(png).decode("ascii")

        return Observation(
            url=self._page.url,
            accessibility_snapshot=ax,
            dom_excerpt=dom,
            screenshot_b64=screenshot_b64,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    async def act(self, action: Action) -> ActionOutcome:
        try:
            if action.type == ActionType.PRESS_KEY:
                if not action.key:
                    return ActionOutcome(ok=False, message="press_key requires 'key'")
                await self._page.keyboard.press(action.key)
                return ActionOutcome(ok=True, message=f"pressed key {action.key!r}")

            if action.type == ActionType.TYPE_TEXT:
                if action.text is None:
                    return ActionOutcome(ok=False, message="type_text requires 'text'")
                for ch in action.text:
                    await self._page.keyboard.press(ch, delay=60)
                return ActionOutcome(ok=True, message=f"typed {len(action.text)} chars")

            if action.type == ActionType.CLICK:
                if action.locator is None:
                    return ActionOutcome(ok=False, message="click requires 'locator'")
                loc = await self._build_locator(action.locator)
                if loc is None:
                    coord_spec = self._find_coordinate_fallback(action.locator)
                    if coord_spec is not None:
                        x_str, y_str = coord_spec.value.split(",")
                        await self._page.mouse.click(float(x_str), float(y_str))
                        return ActionOutcome(ok=True, message="clicked via coordinate fallback")
                    return ActionOutcome(
                        ok=False,
                        message="could not resolve any locator (incl. fallbacks) for click",
                    )
                await loc.click(timeout=5000)
                return ActionOutcome(ok=True, message="clicked")

            if action.type == ActionType.WAIT:
                await self._page.wait_for_timeout(action.wait_ms or 500)
                return ActionOutcome(ok=True, message=f"waited {action.wait_ms or 500}ms")

            return ActionOutcome(ok=False, message=f"unsupported action type {action.type}")
        except Exception as exc:  # noqa: BLE001 — surfaced to caller as a failed outcome
            return ActionOutcome(ok=False, message=f"action raised: {exc}")

    async def resolve(
        self, locator: LocatorSpec, attribute: str = "text", each: bool = False
    ) -> ResolvedField:
        chain = [locator, *locator.fallbacks]
        for spec in chain:
            try:
                result = await self._resolve_one(spec, attribute, each)
                if result.found:
                    result.matched_strategy = spec.strategy
                    return result
            except Exception as exc:  # noqa: BLE001
                continue
        return ResolvedField(found=False, note="no strategy in the locator chain matched")

    async def _resolve_one(
        self, spec: LocatorSpec, attribute: str, each: bool
    ) -> ResolvedField:
        if spec.strategy == LocatorStrategy.ROLE:
            role, name = _parse_role_value(spec.value)
            base = self._page.get_by_role(role, name=name) if name else self._page.get_by_role(role)
            return await self._read(base, spec.position, attribute, each)

        if spec.strategy == LocatorStrategy.TEXT:
            base = self._page.get_by_text(spec.value)
            return await self._read(base, spec.position, attribute, each)

        if spec.strategy == LocatorStrategy.CSS:
            if " >>> " in spec.value:
                outer, inner = spec.value.split(" >>> ", 1)
                outer_loc = _apply_position(self._page.locator(outer), spec.position)
                inner_loc = outer_loc.locator(inner)
                return await self._read(inner_loc, "first", attribute, each=True)
            base = self._page.locator(spec.value)
            return await self._read(base, spec.position, attribute, each)

        return ResolvedField(found=False, note=f"strategy {spec.strategy} is not resolvable for reads")

    async def _read(
        self, locator: Locator, position, attribute: str, each: bool
    ) -> ResolvedField:
        count = await locator.count()
        if count == 0:
            return ResolvedField(found=False, note="0 elements matched")

        targets: list[Locator]
        if each:
            targets = [locator.nth(i) for i in range(count)]
        else:
            targets = [_apply_position(locator, position)]

        values: list[str] = []
        for t in targets:
            values.append(await self._read_attribute(t, attribute))
        return ResolvedField(found=True, values=values)

    async def _read_attribute(self, locator: Locator, attribute: str) -> str:
        if attribute == "text":
            return (await locator.inner_text()).strip()
        val = await locator.get_attribute(attribute)
        return val or ""

    async def _build_locator(self, spec: LocatorSpec) -> Locator | None:
        chain = [spec, *spec.fallbacks]
        for s in chain:
            try:
                if s.strategy == LocatorStrategy.ROLE:
                    role, name = _parse_role_value(s.value)
                    base = (
                        self._page.get_by_role(role, name=name)
                        if name
                        else self._page.get_by_role(role)
                    )
                elif s.strategy == LocatorStrategy.TEXT:
                    base = self._page.get_by_text(s.value)
                elif s.strategy == LocatorStrategy.CSS:
                    base = self._page.locator(s.value)
                else:
                    continue  # COORDINATE has no Locator form; handled by caller
                positioned = _apply_position(base, s.position)
                if await positioned.count() > 0:
                    return positioned
            except Exception:
                continue
        return None

    def _find_coordinate_fallback(self, spec: LocatorSpec) -> LocatorSpec | None:
        for s in [spec, *spec.fallbacks]:
            if s.strategy == LocatorStrategy.COORDINATE:
                return s
        return None

    async def current_url(self) -> str:
        return self._page.url

    async def screenshot_path(self, path: str) -> None:
        await self._page.screenshot(path=path)

    async def close(self) -> None:
        await self._page.context.browser.close() if self._page.context.browser else None
