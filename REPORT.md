# REPORT

## 1. Architecture

**Target and why.** The brief frames this around banking back-office UIs with
no API; I needed a live, permissively-automatable proxy. I initially targeted
NYTimes Wordle as requested, but `nytimes.com/robots.txt` explicitly
disallows automated/AI use ("prohibited... development of any software,
machine learning, artificial intelligence (AI)...") — directly conflicting
with the brief's own ground rule. I checked robots.txt on several public
Wordle clones and picked **hellowordl.net**: `Disallow:` is empty (everything
permitted), the page carries no ad/tracking scripts, and the mechanics
(multi-attempt guess flow, color-coded tile feedback, an invalid-word toast,
win/lose terminal states, on-screen *and* physical-keyboard input) are close
enough to a real business app's "form → validate → confirm" shape to
exercise every core requirement honestly.

**Single-process, stack.** Python + Playwright (async) for the surface,
OpenAI (GPT-4.1, vision + tool-calling) for discovery, Pydantic for the
artifact schema, FastAPI for the operator passthrough, pytest for tests. One
process, no queues/services — the brief explicitly discourages premature
scaling infrastructure, and nothing here needs it yet.

**The seam: `Surface`.** Everything above `surface/base.py`'s `Surface` ABC
(the agent loop, the replay executor) only knows `observe()`/`act()`/
`resolve()`. `PlaywrightWebSurface` is the only implementation; see §4 for
how another surface would slot in.

**Capability granularity — the key design decision.** A recorded/replayed
Capability is not "solve the whole puzzle." hellowordl's answer changes per
session, so a recorded winning guess sequence isn't reusable — exactly the
brief's own point about business outcomes vs. hardcoded happy paths. Instead
the capability is the **atomic interaction**: `submit_guess` — type 5
letters, press Enter, read back per-letter correct/elsewhere/absent plus
game status. This mirrors the brief's own bank example ("look up member,
read balance"): deterministic mechanics, real data back. The LLM's only job
during discovery is the genuinely dynamic part — *which word to guess next*
— which it does by calling low-level tools (`type_letters`, `press_key`,
`read_state`) repeatedly; `agent/distiller.py` then turns the first proven
type→submit sequence in that real transcript into the capability. See §2 for
exactly what's derived from the transcript vs. engineering-authored.

**Discovery tool design.** The model gets no generic `click(x,y)` tool. The
one click-worthy control in this app ("Give up") is destructive, so omitting
a click tool is a structural guarantee against the model reaching for it —
on top of, not instead of, the allowlist guardrail (§6). `type_letters`/
`press_key` send physical key events, which is also just how the real app
works: its on-screen keyboard buttons are `aria-hidden`/`tabindex="-1"`
(decorative), confirmed by probing the live DOM before writing any code.

**The ReAct solver (`agent/solver.py`) — a tool the model calls, not a
replacement for it.** `propose_next_words` deterministically filters a
bundled 8,506-word dictionary down to words consistent with every
`(guess, feedback)` pair so far (`consistent_with_guess` *simulates* the
real duplicate-letter-aware Wordle rule rather than hand-rolling special
cases) and ranks survivors by letter-frequency coverage. The model still
decides — it calls the tool (Reason), then picks a candidate to submit
(Act); nothing is auto-typed. `ToolRunner._do_type` separately rejects,
before a keystroke reaches the browser, anything that isn't a real
dictionary word or that contradicts a known constraint — "invalid word" is
structurally blocked at the tool boundary, not just discouraged. (The
bundled list is a proxy for hellowordl's own dictionary — see §7.)

**Retry/backoff on the LLM call.** Discovery talks to a real, rate-limited
API; live testing surfaced both a dropped-connection error and an explicit
429 across a handful of runs. `_create_completion_with_retry` (`agent/loop.py`)
retries connection/timeout/rate-limit/5xx errors with bounded, jittered
exponential backoff (4 attempts), logging each retry to evidence, and
re-raises anything else immediately rather than retrying into the same wall.

## 2. Artifact schema

`artifact/schema.py`'s `Capability` is the focal point. Every field maps to
a line in §3.2 of the brief:

- **`steps`** — ordered `Step`s (`type_text` / `press_key` / `click` / `wait`).
- **locator + robustness reasoning** — every `LocatorSpec` carries a
  required `robustness_note` and an ordered `fallbacks` chain (ARIA
  role+name first, CSS second, coordinate last resort). The tile-row locator
  is the interesting one: `"table.Game-rows tr.Row:has(td.Row-letter:not(:empty)) >>> td.Row-letter"`
  with `position="last"` — it selects the most recently *filled* row by
  **content**, not a hardcoded row index, so it's correct on attempt 1 or
  attempt 6 and survives whether the prior guess was accepted or rejected. I
  hit this exact bug during validation (see below) and it's now a unit test.
- **typed inputs** — `ParamSpec` (`guess: string`, `pattern="^[A-Za-z]{5}$"`),
  validated before a single step executes.
- **typed outputs** — `ExtractField`s double as the shape declaration *and*
  the extraction instructions, so the schema can't drift from the code that
  reads it: `tile_results` (enum list, per letter), `status_raw` (raw
  status text, for debuggability), `game_status` (enum).
- **checkpoint** — asserts the row's 5 tiles carry a `letter-*` state class
  before trusting anything, polled with a bounded timeout — never assumes a
  keypress worked.
- **error taxonomy** — `invalid_word` / `invalid_length` / `game_already_over`,
  each a `(locator, match_text, category)` triple against the app's own
  `role="alert"` region (the same node its own screen-reader support uses —
  an intentionally stable hook).
- **risk + provenance** — `submit_guess` is `safe`/no-confirmation-required;
  provenance names the exact discovery run, model, and (`reviewed_by: null`)
  that no human has approved it yet — `status: "draft"`, not `"approved"`,
  is the honest state for a freshly-distilled artifact.

**What's derived vs. authored.** `distiller.py` scans the real discovery
transcript for the first successful `type_letters(5 letters)` →
`press_key("Enter")` pair and raises if it never happened — the two `steps`
are literally what the model did. The checkpoint/outputs/error-taxonomy are
engineering-authored, deliberately: replay must not depend on an LLM to
*read* a screenshot every time (that would defeat "no LLM in the decision
loop"), so a human/system author formalizes, once, a deterministic
structured read of exactly the signal the transcript shows the model needed.
This is the same thing a real deployment would do — an engineer hardens a
first discovery run into a production capability rather than trusting
model vision on every future replay.

## 3. Determinism & error handling

Replay (`replay/executor.py`) never calls the LLM. Order of operations, and
why: **(1)** validate inputs against the declared pattern — fail fast before
touching the browser; **(2)** a pre-check against `error_taxonomy` — catches
"the game was already over from a prior call" *before* wasting steps on a
dead session; **(3)** execute steps, each with one bounded retry for
transient conditions; **(4)** poll the checkpoint **and** the error taxonomy
together, every tick (up to `timeout_ms`, 150ms interval); **(5)** if the
checkpoint never holds and nothing in the taxonomy explains it, that's an
undiagnosed hard failure; **(6)** only on checkpoint success, extract
outputs.

**A real ~200x latency bug this caught, live.** Step (4) originally checked
the checkpoint alone for the full timeout, and only consulted the taxonomy
*after* it expired — so every rejected guess (e.g. "not a valid word") sat
through a silent, unexplained 4-second wait before resolving, even though
the app's own rejection signal (its alert region) appears almost instantly.
I measured this against the live site — 4.02s from submit to resolution,
timestamps in the evidence log — before fixing `_poll_checkpoint_or_taxonomy`
to check both on every tick, not checkpoint-then-taxonomy-after-timeout. Same
measurement after the fix: **~20ms**. A genuinely slow-but-real success
(still animating, nothing in the taxonomy matches) is unaffected — polling
continues exactly as before until the checkpoint holds or the timeout
expires; only the "this was never going to succeed" case got faster.

This ordering is what keeps `ReplayResult`'s three cases honest
(`replay/results.py`):
- **`Success`** — checkpoint held, outputs extracted.
- **`BusinessOutcome`** — a real, expected app response (`invalid_word`,
  `invalid_length`, `game_already_over`) — the caller needs this, not an
  exception.
- **`Failure`** — checkpoint never held *and* nothing in the taxonomy
  explains it — carries `step_id`/`expected`/`observed`/`message`, and
  triggers `evidence.save_failure_bundle()` (screenshot + full AX snapshot +
  DOM excerpt) for debugging without re-running.

**A real bug this caught.** My first checkpoint implementation checked "is
the tile's `class` attribute non-empty" — but every tile carries the base
class `Row-letter` even before evaluation, so an invalid-word rejection
(which leaves the base class in place, no `letter-*` suffix) still read as
"non-empty" and the executor reported `success` with garbage outputs. I
caught this by actually replaying against the live site (see
`/evidence/replay-error-...`), fixed the condition to check for the
`letter-` substring specifically, and it's now `test_replay_invalid_word_is_business_outcome_not_a_crash`.
Determinism claims that were only tested against a fake surface would have
missed this entirely — the live validation runs in `README.md`'s demo path
are load-bearing, not decorative.

**Multi-invocation correctness** was also validated live: 3 sequential
`submit_guess` calls against one session (wrong, wrong, correct) correctly
targeted row 0 → row 1 → row 2 with no index bookkeeping needed, then a 4th
call correctly short-circuited as `game_already_over` without touching the
keyboard.

## 4. Heterogeneity & multi-tenant (design, not built)

**Surface abstraction.** Nothing above `Surface.observe()`/`act()` knows
Playwright exists. A **legacy web** surface would keep the same ABC and
lean on `LocatorStrategy.TEXT`/table-position CSS instead of ARIA roles
(legacy server-rendered apps rarely have roles/test IDs, but usually have
stable label text and table layout). A **desktop** surface would implement
the same ABC over an OS accessibility API (e.g. AXUIElement on macOS,
UI Automation on Windows) instead of Playwright, returning the same
`Observation`/`ActionOutcome` shapes. `Capability`, `ReplayExecutor`, and
`GuardedSurface` would not change — only `LocatorStrategy` choices and one
new `Surface` implementation.

**Multi-tenant reuse.** I'd extend `TargetSpec` with an `app_fingerprint`
(vendor product + version signature, e.g. a hash of a stable DOM/AX
landmark set) and add a `tenant_overrides: dict[tenant_id, list[LocatorSpec override]]`
alongside the base `Capability`. Replay resolves a tenant's override chain
first, falls back to the base capability's locators — so one recording
generalizes across tenants running the same underlying product, and a
tenant with a themed/relabeled build only needs a small override, not a
re-record. **Drift detection**: every replay's checkpoint pass/fail is
already a signal; I'd aggregate it per `(capability_id, tenant_id)` and flag
a capability for review once its checkpoint-failure rate crosses a
threshold for a specific tenant — the stretch goal "confidence & approval"
(draft→approved, `status` field already exists in the schema for exactly
this) is the natural next layer on top, not built here because the brief
asks for at most one or two stretch goals and I prioritized depth on the
core loop instead.

## 5. Escalation & handoff

**Detecting stuck.** Two trigger paths converge on the same handling code in
`agent/loop.py`: the discovery loop calling `escalate()` itself (a tool it's
told to use when genuinely stuck), and `max_steps`/timeout exhaustion without
`finish_goal` (a forced "dead end"). A third — `GuardedSurface` blocking a
risky/irreversible action (§6) — is demonstrated standalone by the
`escalate-demo` CLI command, since the discovery agent has no click tool to
trigger it for real (§1).

**Control-transfer model.** `EscalationManager` is an explicit state
machine: `control_owner ∈ {automation, human}`. On escalation it captures a
full intervention bundle (reason, screenshot, AX snapshot,
`intervention_request.json`) and flips ownership — critically, **the same
Playwright page/context stays open**; nothing is torn down or recreated. A
human then acts on that exact session either by touching the visible headed
browser window directly, or through `escalation/operator_server.py`'s tiny
FastAPI passthrough (`POST /act` dispatches straight onto the live page;
`GET /screenshot` re-captures it live). `POST /resume` flips ownership back.
Every human action taken while `control_owner == "human"` is logged through
the same evidence pipeline as automated steps (`human_actions.json`) — "what
did the human do" is part of the run record, not a gap in it.

**The loop actually resumes reasoning, not just the mechanism.** When
`run_discovery` sees `tool_runner.escalated`, it calls
`request_intervention`, blocks on `wait_for_resume()`, then — since the
human acted directly on the raw surface via the operator server, bypassing
`ToolRunner` entirely — re-derives the solver's guess history from scratch
by scanning every row of the live board
(`ToolRunner.resync_guess_history_from_board`) rather than trusting stale
state. It appends one message summarizing what happened and `continue`s the
same `while` loop with a fresh step budget — the model picks up with no
special-casing, same tools, now-accurate `propose_next_words` output.

**Validated live, for real** (`/evidence/discovery-20260816T153845Z/`): I ran
discovery with a tiny step budget to force a genuine dead-end escalation
after one guess (`AROSE`). While it waited, I acted as the human operator —
hitting the exact REST endpoints `operator.html`'s JS would call — checked
`/state`, fetched `/screenshot`, typed a second guess (`SHINE`) via `/act`,
then `/resume`d. The model's next tool call, `propose_next_words`, returned
51 candidates — correctly reflecting *both* rows, proving the resync read
the real board, not stale state. It kept reasoning, hit the budget wall two
more times (each a genuine dead-end, each resumed — one with a further human
action, one with none), narrowed to one candidate, and won guessing `DITTO`.
Four escalation cycles, one real human intervention, one evidence bundle
covering discovery, the solver, and escalation together.

**What's mocked, deliberately (per the brief's own scope note).** The
operator UI (`operator.html`) is a bare static page — a screenshot, a
reason, three buttons, 1.5s polling. No websockets, no real-time
co-browsing, no auth. `/evidence/escalation-20260815T183047Z/` is a second,
earlier example of the *other* trigger path (a blocked risky click, not a
dead end) using the same mechanism.

## 6. Safety

**Allowlist** (`config/allowlist.yaml`, enforced by `guardrails/allowlist.py`'s
`GuardedSurface` — a decorator wrapping every real `Surface`, so there is
exactly one enforcement point, not scattered checks): an explicit domain
allowlist (`hellowordl.net` only, deny-by-default otherwise), an explicit
action-type allowlist (`press_key`/`type_text`/`click`/`wait` — no
`navigate` inside the guarded loop), and a risk classifier.

**Risky/irreversible handling: block, not confirm-and-proceed.** Anything
matching `risky_click_names` (currently just "Give up", which discards
in-progress game state — the proxy-domain stand-in for something like
"submit disbursement" in a real banking flow) is **always blocked** from
unattended execution; there is no in-band override the automation itself
can use. I chose blocking over auto-confirmation or silent-flag-and-proceed
because an irreversible action in a regulated context should never be a
default outcome of an LLM's judgment call — only a human, via the
escalation path in §5, can actually take that action. This is defense in
depth on top of the tool-design choice in §1 (the discovery agent isn't even
given a click tool that could reach "Give up").

**Redaction** (`guardrails/redaction.py`) strips secret-shaped strings
(API keys, bearer tokens, emails, card/SSN-shaped numbers) from *everything*
written to evidence or artifacts — `EvidenceLogger` and `artifact/store.py`
both route through it. Wordle itself has no real PII, so this is exercised
by unit tests with synthetic secrets rather than real domain data; in the
banking target this is the seam that matters (account numbers, session
tokens, auth headers visible in a debug panel), and the pattern list is
where that would grow.

**A second, non-safety guardrail worth naming.** `ToolRunner` enforces a
*correctness* gate alongside the allowlist: `type_letters` rejects anything
that isn't a real dictionary word, or that contradicts a known constraint,
before it reaches the browser (§1). Not part of `GuardedSurface`'s policy —
tool-level input validation — but the same instinct: block bad input at the
boundary.

**Limits, honestly.** The allowlist is a static YAML file, not per-capability
policy tied to a review workflow; redaction is pattern-based (regex), which
will miss anything that doesn't look like a known secret shape — a real
deployment would need structured PII detection, not just regex, and a
human-reviewed allowlist per tenant/app rather than one global file. The
dictionary guardrail is also imperfect in one specific way — see §7.

## 7. Cuts

Left out, deliberately, with what I'd build next:

- **The bundled dictionary is a proxy, not hellowordl's actual word list.**
  `agent/data/words5.txt` is filtered from macOS's system dictionary, which
  includes archaic/rare words hellowordl almost certainly doesn't recognize,
  and may miss some it does accept. So `type_letters`'s guarantee is
  precise but narrower than it sounds: "not a real word *per our list*" is
  blocked pre-browser, but a word we consider valid could still occasionally
  be rejected live — handled gracefully (the existing `invalid_word`
  business outcome), not a crash, but worth stating exactly. A production
  version would source the real target word list where one exists.
- **Multi-tenant/legacy/desktop surfaces** — designed in §4, not built, per
  the brief's own instruction not to prematurely build scaling
  infrastructure.
- **Stretch goals** — I picked none over depth on the core loop, but the
  schema leaves room: `Capability.status` (`draft`/`approved`) is already
  there for "confidence & approval" gating; `app_fingerprint`/
  `tenant_overrides` (§4) is the seam for cross-tenant reuse.
- **A second, independent `read_board_state`/`check_game_status` capability**
  — `submit_guess` already returns full tile feedback and game status on
  every call, so a separate read-only capability would be redundant for
  this proxy domain; in a system with more read-heavy flows (e.g. "look up
  member" without a subsequent write) it would earn its place.
- **Multi-run flakiness scoring** — the executor does one bounded retry per
  step and the LLM call has its own bounded retry (§1); I did not build the
  stretch-goal-level "replay N times, report a stability signal," though the
  evidence from `/evidence/` (multiple live replay calls, zero flakiness
  observed) is a reasonable proxy for a game this stable.
- **Retry policy is fixed, not configurable per call site** — the same
  backoff schedule (4 attempts, capped exponential + jitter) applies to
  every discovery LLM call; a production system serving multiple capability
  types would likely want this tunable per model/provider rather than a
  single constant in `agent/loop.py`.
