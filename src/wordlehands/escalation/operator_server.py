"""A deliberately bare operator UI (Section 3.6 scope note: "a full real-time
co-browsing operator console is out of scope... mock the operator UI if
needed, but make the handoff mechanism and control-transfer model real").

What's mocked: the UI (`operator.html` — a static polling page, no
websockets, no fancy co-browsing). What's real: `/act` dispatches straight
onto the SAME live Playwright page the automation was using, and `/resume`
flips the same control_owner state machine the automation checks — a human
using this page (or just clicking into the visible headed browser window
directly) is genuinely operating the live session, not a mock of one.
"""

from __future__ import annotations

import base64
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from wordlehands.escalation.manager import EscalationManager
from wordlehands.surface.base import Action, ActionType

OPERATOR_HTML_PATH = Path(__file__).parent / "operator.html"


class ActRequest(BaseModel):
    type: str  # "press_key" | "type_text"
    key: str | None = None
    text: str | None = None


def build_app(manager: EscalationManager) -> FastAPI:
    app = FastAPI(title="wordlehands operator console (mocked UI, real control transfer)")

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return OPERATOR_HTML_PATH.read_text()

    @app.get("/state")
    async def state():
        return {
            "control_owner": manager.control_owner,
            "reason": manager.reason,
            "human_action_count": len(manager.human_actions),
        }

    @app.get("/screenshot")
    async def screenshot():
        obs = await manager.surface.observe(include_screenshot=True)
        png = base64.b64decode(obs.screenshot_b64)
        return Response(content=png, media_type="image/png")

    @app.post("/act")
    async def act(req: ActRequest):
        if manager.control_owner != "human":
            return {"ok": False, "message": "control is not currently with the human operator"}
        if req.type == "press_key":
            outcome = await manager.surface.act(
                Action(type=ActionType.PRESS_KEY, key=req.key, reason="human operator action")
            )
        elif req.type == "type_text":
            outcome = await manager.surface.act(
                Action(type=ActionType.TYPE_TEXT, text=req.text, reason="human operator action")
            )
        else:
            return {"ok": False, "message": f"unsupported action type {req.type}"}
        await manager.record_human_action(
            f"{req.type} {req.key or req.text}", outcome.ok, outcome.message
        )
        return {"ok": outcome.ok, "message": outcome.message}

    @app.post("/resume")
    async def resume():
        await manager.resume()
        return {"ok": True}

    return app
