# wordlehands

A computer-use automation system: an LLM discovers how to play a live,
no-API web game, the interaction it discovers is distilled into a typed,
versioned **Capability** artifact, that artifact replays **deterministically**
(no LLM) with structured success / business-outcome / failure results, and a
human can be escalated to and take control of the **same live session** when
the system can't safely proceed. Built for interface.ai's take-home
(`/REPORT.md` has the full design write-up — architecture, schema, error
handling, escalation, safety, and what was cut).

**Target surface:** [hellowordl.net](https://hellowordl.net) (a Wordle-style
word-guessing game), *not* NYTimes Wordle — `nytimes.com/robots.txt`
explicitly disallows automated/AI use, which conflicts with the project's
own ground rules. hellowordl.net has a fully permissive `robots.txt`
(`Disallow:` empty) and real enough mechanics (multi-attempt flow, on-screen
+ physical keyboard input, color-coded tile feedback, an invalid-word toast,
win/lose states) to exercise every requirement. See REPORT.md for the full
reasoning.

## Setup

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and an OpenAI API
key (used only for the discovery step — replay never calls an LLM).

```bash
uv sync
uv run playwright install chromium
cp .env.example .env   # then put your real key in .env: OPENAI_API_KEY=sk-...
```

`.env` is gitignored. Optional overrides (see `.env.example`):
`WORDLEHANDS_MODEL` (default `gpt-4.1`), `WORDLEHANDS_TARGET_URL` (default
`https://hellowordl.net/`), `WORDLEHANDS_HEADLESS` (`1` for headless, default
headed).

Run the test suite (no browser, no API key, no network needed):

```bash
uv run pytest tests/ -v
```

## Demo path

**1. Discovery** — a real GPT-4.1 run plays a live game via the tools in
`agent/tools.py` (physical keystrokes + reads), then a successful run is
distilled into a saved Capability artifact:

```bash
uv run python -m wordlehands.cli discover \
  --goal "Play hello wordl (a Wordle-style game) and win by guessing the 5-letter target word within 6 attempts." \
  --headless --max-steps 30
```

Writes `/artifacts/submit_guess.v1.0.0.json` and a full evidence bundle
(step log, screenshots, tool-call trace, result) to
`/evidence/discovery-<timestamp>/`. A completed example run is committed at
`/evidence/discovery-20260815T182917Z/` — the agent won in 3 guesses
(CRANE → PLUSH → SHUSH).

**2. Deterministic replay** — no LLM involved; drives the browser using only
the artifact's recorded locators:

```bash
# success path
uv run python -m wordlehands.cli replay \
  --capability artifacts/submit_guess.v1.0.0.json \
  --input '{"guess":"crane"}' --seed 100 --headless

# business-outcome path: not a real dictionary word — a legitimate result, not a crash
uv run python -m wordlehands.cli replay \
  --capability artifacts/submit_guess.v1.0.0.json \
  --input '{"guess":"zzzzq"}' --seed 101 --headless
```

`--seed` pins hellowordl's target word (`?seed=N`) so a run is exactly
reproducible. Each invocation prints a structured `ReplayResult`
(`success` / `business_outcome` / `failure`) and writes evidence to
`/evidence/replay-<timestamp>/`. Example runs are committed at
`/evidence/replay-success-20260815T183008Z/` and
`/evidence/replay-error-20260815T183012Z/`.

**3. Escalation & handoff** — forces a blocked risky action (the guardrail
refuses to click "Give up"), pauses, and exposes the *same live session* for
manual control:

```bash
uv run python -m wordlehands.cli escalate-demo --headless --port 8765
```

Prints an operator URL (`http://127.0.0.1:8765`) with a deliberately bare
mock console (screenshot + reason + type/press/resume controls). Open it in
a browser, or drive the same REST it uses directly:

```bash
curl -s -X POST http://127.0.0.1:8765/act -H "Content-Type: application/json" -d '{"type":"type_text","text":"shard"}'
curl -s -X POST http://127.0.0.1:8765/act -H "Content-Type: application/json" -d '{"type":"press_key","key":"Enter"}'
curl -s -X POST http://127.0.0.1:8765/resume
```

`/act` dispatches straight onto the same live Playwright page; `/resume`
flips control back and the CLI process continues and exits. Evidence
(`intervention_request.json`, before/after screenshots, `human_actions.json`)
is written to `/evidence/escalation-<timestamp>/` — a committed example at
`/evidence/escalation-20260815T183047Z/` shows the human's guess landing in
row 2 directly under automation's row-1 guess, on the same board.

## Project layout

```
src/wordlehands/
  surface/      Surface ABC + the only concrete impl (Playwright web)
  agent/        discovery loop (OpenAI tool-calling), tool palette, distiller
  artifact/     the Capability schema (pydantic) + versioned JSON store
  replay/       deterministic executor + the Success/BusinessOutcome/Failure result contract
  guardrails/   domain/action-type allowlist + risk classification + redaction
  escalation/   control-transfer state machine + a tiny FastAPI operator passthrough
  evidence/     structured JSONL logging + screenshots + failure bundles
  cli.py        discover / replay / escalate-demo
config/allowlist.yaml   the policy discover/replay/escalate-demo all enforce
artifacts/      saved Capability artifacts
evidence/       one committed example run per capability (discovery, replay x2, escalation)
tests/          pytest, browser-free (a FakeSurface stands in for Playwright)
```

## What's mocked, and why

- **Operator console UI** is a bare static polling page (`escalation/operator.html`), not a
  real-time co-browsing console — explicitly in-scope-to-mock per the brief. The
  *mechanism* underneath it (same-session control transfer via `/act`, `/resume`) is real.
- **Multi-tenant reuse and a legacy/desktop surface** are design-only (REPORT.md §4) —
  the brief asks for a credible design, not an implementation, and says not to build
  scaling infrastructure prematurely.

See `/REPORT.md` → "Cuts" for the full list and what would come next.
