import pytest

from tests.fake_surface import FakeSurface
from wordlehands.agent.tools import ToolRunner
from wordlehands.evidence.logger import EvidenceLogger
from wordlehands.surface.base import (
    Action,
    ActionOutcome,
    LocatorStrategy,
    Observation,
    ResolvedField,
    Surface,
)


@pytest.fixture
def evidence(tmp_path):
    return EvidenceLogger(tmp_path / "run")


class _RowAwareFakeSurface(Surface):
    """Unlike FakeSurface (which always answers for 'the last row'), this
    differentiates by row index — needed to test resync_guess_history_from_board,
    which reads the board row-by-row rather than 'most recently filled'.
    """

    def __init__(self, rows: list[tuple[str, list[str]]]):
        self._rows = rows  # already-evaluated (word, states) pairs, in board order

    async def observe(self, include_screenshot: bool = True) -> Observation:
        return Observation(
            url="https://hellowordl.net/",
            accessibility_snapshot="",
            dom_excerpt="",
            screenshot_b64=None,
            timestamp="2026-01-01T00:00:00Z",
        )

    async def act(self, action: Action) -> ActionOutcome:
        return ActionOutcome(ok=True, message="ok")

    async def resolve(self, locator, attribute: str = "text", each: bool = False) -> ResolvedField:
        if locator.strategy != LocatorStrategy.CSS or "Row-letter" not in locator.value:
            return ResolvedField(found=False)
        idx = locator.position
        if not isinstance(idx, int) or idx >= len(self._rows):
            return ResolvedField(found=False, note="row not filled")
        word, states = self._rows[idx]
        if attribute == "text":
            return ResolvedField(found=True, values=list(word))
        if attribute == "class":
            return ResolvedField(found=True, values=[f"Row-letter letter-{s}" for s in states])
        return ResolvedField(found=False)

    async def current_url(self) -> str:
        return "https://hellowordl.net/"

    async def screenshot_path(self, path: str) -> None:
        pass

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_type_letters_rejects_non_dictionary_word_without_touching_surface(evidence):
    surface = FakeSurface(mode="success")
    runner = ToolRunner(surface, evidence)

    result = await runner.dispatch("type_letters", {"letters": "zzzzq"})

    import json

    parsed = json.loads(result)
    assert parsed["ok"] is False
    assert surface.actions == []  # never reached the browser


@pytest.mark.asyncio
async def test_type_letters_accepts_real_word_and_dispatches_to_surface(evidence):
    surface = FakeSurface(mode="success")
    runner = ToolRunner(surface, evidence)

    result = await runner.dispatch("type_letters", {"letters": "crane"})

    import json

    parsed = json.loads(result)
    assert parsed["ok"] is True
    assert len(surface.actions) == 1


@pytest.mark.asyncio
async def test_pressing_enter_captures_real_tile_feedback_into_history(evidence):
    surface = FakeSurface(mode="success")  # tiles: correct, elsewhere, absent, absent, correct
    runner = ToolRunner(surface, evidence)

    await runner.dispatch("type_letters", {"letters": "crane"})
    await runner.dispatch("press_key", {"key": "Enter"})

    assert runner._guess_history == [
        ("crane", ["correct", "elsewhere", "absent", "absent", "correct"])
    ]


@pytest.mark.asyncio
async def test_type_letters_rejects_word_contradicting_prior_feedback(evidence):
    surface = FakeSurface(mode="success")
    runner = ToolRunner(surface, evidence)
    await runner.dispatch("type_letters", {"letters": "crane"})
    await runner.dispatch("press_key", {"key": "Enter"})
    actions_after_first_guess = len(surface.actions)

    # "irate" contradicts the captured feedback (c/r/a were correct/elsewhere,
    # not absent as "irate" would require for those letters) — must be blocked.
    import json

    result = await runner.dispatch("type_letters", {"letters": "irate"})
    parsed = json.loads(result)

    assert parsed["ok"] is False
    assert len(surface.actions) == actions_after_first_guess  # no new action reached the surface


@pytest.mark.asyncio
async def test_propose_next_words_returns_candidates_consistent_with_history(evidence):
    surface = FakeSurface(mode="success")
    runner = ToolRunner(surface, evidence)
    await runner.dispatch("type_letters", {"letters": "crane"})
    await runner.dispatch("press_key", {"key": "Enter"})

    import json

    result = json.loads(await runner.dispatch("propose_next_words", {}))

    assert "candidates" in result
    assert all(len(w) == 5 for w in result["candidates"])
    assert result["remaining_possible_count"] >= len(result["candidates"])


@pytest.mark.asyncio
async def test_resync_guess_history_from_board_reads_every_filled_row(evidence):
    rows = [
        ("crane", ["absent", "absent", "absent", "correct", "correct"]),
        ("shine", ["elsewhere", "absent", "correct", "absent", "correct"]),
    ]
    surface = _RowAwareFakeSurface(rows)
    runner = ToolRunner(surface, evidence)
    runner._guess_history = [("stale", ["absent"] * 5)]  # simulate pre-handoff staleness

    await runner.resync_guess_history_from_board()

    assert runner._guess_history == rows  # fully replaced, stops at the first empty row


@pytest.mark.asyncio
async def test_resync_guess_history_from_board_handles_no_rows_filled_yet(evidence):
    surface = _RowAwareFakeSurface(rows=[])
    runner = ToolRunner(surface, evidence)

    await runner.resync_guess_history_from_board()

    assert runner._guess_history == []
