from __future__ import annotations

import asyncio
import json
from pathlib import Path

import click
from playwright.async_api import async_playwright

from wordlehands.agent.distiller import NoDiscoveredSubmission, distill_submit_guess_capability
from wordlehands.agent.loop import run_discovery
from wordlehands.artifact import store
from wordlehands.config import settings
from wordlehands.escalation.manager import EscalationManager
from wordlehands.escalation.operator_server import build_app
from wordlehands.evidence.logger import new_run
from wordlehands.guardrails.allowlist import AllowlistPolicy, GuardedSurface
from wordlehands.replay.executor import ReplayExecutor
from wordlehands.surface.base import Action, ActionType, LocatorSpec, LocatorStrategy
from wordlehands.surface.playwright_web import PlaywrightWebSurface


async def _new_surface(headless: bool, target_url: str):
    pw = await async_playwright().start()
    browser = await pw.chromium.launch(headless=headless)
    page = await browser.new_page(viewport={"width": 1280, "height": 900})
    await page.goto(target_url, wait_until="networkidle")
    surface = PlaywrightWebSurface(page)
    return pw, browser, surface


def _guarded(raw_surface, evidence):
    policy = AllowlistPolicy.load(settings.allowlist_path)
    return GuardedSurface(
        raw_surface,
        policy,
        on_violation=lambda a, r: evidence.log("guardrail_blocked", action=a.type.value, reason=r),
    )


@click.group()
def cli():
    """wordlehands — computer-use automation system, demonstrated against hellowordl.net."""


@cli.command()
@click.option("--goal", required=True, help="Natural-language goal for the discovery agent.")
@click.option("--target-url", default=None)
@click.option("--headless/--headed", default=None)
@click.option("--max-steps", default=40)
def discover(goal: str, target_url: str | None, headless: bool | None, max_steps: int):
    """Run a real LLM-driven discovery loop against the live target, then
    distill a successful run into a saved Capability artifact."""
    asyncio.run(_discover(goal, target_url, headless, max_steps))


async def _discover(goal, target_url, headless, max_steps):
    target_url = target_url or settings.target_base_url
    headless = settings.headless if headless is None else headless

    pw, browser, raw_surface = await _new_surface(headless, target_url)
    evidence = new_run(settings.evidence_dir, "discovery")
    guarded = _guarded(raw_surface, evidence)

    step_count = 0

    def _show_action(tool: str, args: dict, ok: bool, message: str) -> None:
        nonlocal step_count
        step_count += 1
        status = "ok" if ok else "FAILED"
        click.echo(f"  [{step_count:02d}] {tool}({json.dumps(args)}) -> {status}: {message}")

    try:
        click.echo(f"Discovery agent starting — goal: {goal}\n")
        tool_runner = await run_discovery(
            goal, target_url, guarded, evidence, max_steps=max_steps, on_call=_show_action
        )
        evidence.write_json("discovery_calls.json", {"calls": tool_runner.calls})

        result = {
            "finished": tool_runner.finished,
            "finish_result": tool_runner.finish_result,
            "escalated": tool_runner.escalated,
            "escalate_reason": tool_runner.escalate_reason,
        }

        if tool_runner.finished:
            click.echo(f"\nDiscovery finished: {tool_runner.finish_result}")
            try:
                cap = distill_submit_guess_capability(
                    tool_runner.calls, evidence.run_dir.name, settings.openai_model, target_url
                )
                path = store.save(cap, settings.artifacts_dir)
                evidence.log("artifact_saved", path=str(path))
                result["artifact_path"] = str(path)
                click.echo(f"Saved capability artifact: {path}")
            except NoDiscoveredSubmission as exc:
                click.echo(f"Could not distill a capability: {exc}")
        else:
            click.echo(f"Discovery escalated: {tool_runner.escalate_reason}")

        evidence.write_result(result)
    finally:
        await browser.close()
        await pw.stop()

    click.echo(f"Evidence: {evidence.run_dir}")


@cli.command()
@click.option("--capability", "capability_path", required=True, type=click.Path(exists=True))
@click.option("--input", "input_json", required=True, help='JSON input params, e.g. \'{"guess":"crane"}\'')
@click.option("--target-url", default=None)
@click.option("--headless/--headed", default=None)
@click.option("--seed", default=None, help="Optional ?seed= value for a deterministic target word.")
def replay(capability_path, input_json, target_url, headless, seed):
    """Replay a saved Capability artifact deterministically — no LLM involved."""
    asyncio.run(_replay(capability_path, input_json, target_url, headless, seed))


async def _replay(capability_path, input_json, target_url, headless, seed):
    cap = store.load(Path(capability_path))
    inputs = json.loads(input_json)
    target_url = target_url or cap.target.base_url
    if seed:
        sep = "&" if "?" in target_url else "?"
        target_url = f"{target_url}{sep}seed={seed}"
    headless = settings.headless if headless is None else headless

    pw, browser, raw_surface = await _new_surface(headless, target_url)
    evidence = new_run(settings.evidence_dir, "replay")
    guarded = _guarded(raw_surface, evidence)

    try:
        executor = ReplayExecutor(guarded, evidence)
        result = await executor.run(cap, inputs)
        click.echo(json.dumps(result.model_dump(), indent=2, default=str))
    finally:
        await browser.close()
        await pw.stop()

    click.echo(f"Evidence: {evidence.run_dir}")


@cli.command(name="escalate-demo")
@click.option("--target-url", default=None)
@click.option("--headless/--headed", default=False)
@click.option("--port", default=8765)
def escalate_demo(target_url, headless, port):
    """Force a stuck state (a blocked risky action) and demonstrate the full
    pause -> human takes control of the SAME live session -> resume handoff."""
    asyncio.run(_escalate_demo(target_url, headless, port))


async def _escalate_demo(target_url, headless, port):
    import uvicorn

    target_url = target_url or settings.target_base_url
    pw, browser, raw_surface = await _new_surface(headless, target_url)
    evidence = new_run(settings.evidence_dir, "escalation")
    guarded = _guarded(raw_surface, evidence)

    await guarded.act(Action(type=ActionType.TYPE_TEXT, text="crane", reason="demo: agent's normal first guess"))
    await guarded.act(Action(type=ActionType.PRESS_KEY, key="Enter", reason="demo: agent submits guess"))
    await asyncio.sleep(1.0)

    give_up_locator = LocatorSpec(
        strategy=LocatorStrategy.ROLE,
        value="role=button;name=Give up",
        position="first",
        robustness_note="ARIA role+accessible name for the Give Up control.",
    )
    blocked = await guarded.act(
        Action(type=ActionType.CLICK, locator=give_up_locator, reason="demo: agent decides it is stuck and wants to give up")
    )
    click.echo(
        f"Guarded action result: ok={blocked.ok} blocked_by_guardrail={blocked.blocked_by_guardrail} message={blocked.message}"
    )

    manager = EscalationManager(raw_surface, evidence)
    await manager.request_intervention(
        reason=(
            "Automation attempted a risky/irreversible action ('Give up') that is blocked by "
            "policy; a human must decide how to proceed."
        ),
        capability_id="submit_guess",
    )

    app = build_app(manager)
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    server_task = asyncio.create_task(server.serve())

    click.echo(f"\nEscalated. Operator console: http://127.0.0.1:{port}")
    click.echo(
        "Open it in a browser and take manual action on the live session (or interact with "
        "the visible browser window directly), then click 'Resume automation'."
    )
    click.echo("Waiting for resume...\n")

    await manager.wait_for_resume()
    server.should_exit = True
    await server_task

    obs = await raw_surface.observe()
    if obs.screenshot_b64:
        evidence.save_screenshot_b64(obs.screenshot_b64, "after_resume.png")
    evidence.log("automation_resumed", url=obs.url)
    evidence.write_result({"status": "escalation_demo_complete", "human_actions": manager.human_actions})

    await browser.close()
    await pw.stop()
    click.echo(f"Done. Evidence: {evidence.run_dir}")


if __name__ == "__main__":
    cli()
