import pytest

from wordlehands.agent.distiller import distill_submit_guess_capability
from wordlehands.evidence.logger import EvidenceLogger
from wordlehands.replay.executor import InputValidationError, ReplayExecutor

from .fake_surface import FakeSurface

FAKE_DISCOVERY_CALLS = [
    {"tool": "read_state", "args": {}, "ok": True, "message": ""},
    {"tool": "type_letters", "args": {"letters": "crane"}, "ok": True, "message": "typed 5 chars"},
    {"tool": "press_key", "args": {"key": "Enter"}, "ok": True, "message": "pressed key 'Enter'"},
]


@pytest.fixture
def capability():
    return distill_submit_guess_capability(
        FAKE_DISCOVERY_CALLS,
        discovery_run_id="test-run",
        model_used="test-model",
        target_url="https://hellowordl.net/",
    )


@pytest.fixture
def evidence(tmp_path):
    return EvidenceLogger(tmp_path / "run")


async def test_replay_success_extracts_typed_outputs(capability, evidence):
    surface = FakeSurface(mode="success")
    result = await ReplayExecutor(surface, evidence).run(capability, {"guess": "crane"})

    assert result.status == "success"
    assert result.outputs["tile_results"] == ["correct", "elsewhere", "absent", "absent", "correct"]
    assert len(surface.actions) == 2  # type_text + press_key, nothing else


async def test_replay_invalid_word_is_business_outcome_not_a_crash(capability, evidence):
    surface = FakeSurface(mode="invalid_word")
    result = await ReplayExecutor(surface, evidence).run(capability, {"guess": "zzzzq"})

    assert result.status == "business_outcome"
    assert result.code == "invalid_word"


async def test_replay_precondition_short_circuits_when_game_already_over(capability, evidence):
    surface = FakeSurface(mode="game_already_over")
    result = await ReplayExecutor(surface, evidence).run(capability, {"guess": "crane"})

    assert result.status == "business_outcome"
    assert result.code == "game_already_over"
    assert surface.actions == []  # no steps were executed at all


async def test_replay_hard_failure_when_nothing_explains_checkpoint_miss(capability, evidence):
    surface = FakeSurface(mode="hard_failure")
    result = await ReplayExecutor(surface, evidence).run(capability, {"guess": "crane"})

    assert result.status == "failure"
    assert result.message
    assert (evidence.run_dir / "failure" / "reason.txt").exists()


async def test_replay_rejects_input_not_matching_declared_pattern(capability, evidence):
    surface = FakeSurface(mode="success")
    with pytest.raises(InputValidationError):
        await ReplayExecutor(surface, evidence).run(capability, {"guess": "toolong"})


async def test_replay_rejects_missing_required_input(capability, evidence):
    surface = FakeSurface(mode="success")
    with pytest.raises(InputValidationError):
        await ReplayExecutor(surface, evidence).run(capability, {})
