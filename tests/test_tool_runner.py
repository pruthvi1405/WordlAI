import pytest

from tests.fake_surface import FakeSurface
from wordlehands.agent.tools import ToolRunner
from wordlehands.evidence.logger import EvidenceLogger


@pytest.fixture
def evidence(tmp_path):
    return EvidenceLogger(tmp_path / "run")


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
