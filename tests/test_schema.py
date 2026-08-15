import pytest

from wordlehands.agent.distiller import NoDiscoveredSubmission, distill_submit_guess_capability
from wordlehands.artifact.schema import Capability

GOOD_CALLS = [
    {"tool": "type_letters", "args": {"letters": "crane"}, "ok": True, "message": ""},
    {"tool": "press_key", "args": {"key": "Enter"}, "ok": True, "message": ""},
]


def test_distilled_capability_round_trips_through_json():
    cap = distill_submit_guess_capability(GOOD_CALLS, "run-1", "test-model", "https://hellowordl.net/")
    dumped = cap.model_dump_json()
    reloaded = Capability.model_validate_json(dumped)
    assert reloaded == cap


def test_distilled_capability_declares_typed_input_with_pattern():
    cap = distill_submit_guess_capability(GOOD_CALLS, "run-1", "test-model", "https://hellowordl.net/")
    (guess_param,) = cap.inputs
    assert guess_param.name == "guess"
    assert guess_param.pattern == "^[A-Za-z]{5}$"


def test_distilled_capability_has_checkpoint_and_error_taxonomy():
    cap = distill_submit_guess_capability(GOOD_CALLS, "run-1", "test-model", "https://hellowordl.net/")
    assert cap.checkpoint is not None
    codes = {e.code for e in cap.error_taxonomy}
    assert {"invalid_word", "invalid_length", "game_already_over"} <= codes


def test_distilling_without_a_real_submission_in_the_transcript_raises():
    calls_without_submission = [
        {"tool": "read_state", "args": {}, "ok": True, "message": ""},
        {"tool": "type_letters", "args": {"letters": "cr"}, "ok": True, "message": ""},  # not 5 letters
    ]
    with pytest.raises(NoDiscoveredSubmission):
        distill_submit_guess_capability(calls_without_submission, "run-1", "test-model", "https://hellowordl.net/")


def test_risky_action_is_flagged_irreversible_in_write_up_sense():
    cap = distill_submit_guess_capability(GOOD_CALLS, "run-1", "test-model", "https://hellowordl.net/")
    assert cap.risk.level == "safe"
    assert cap.risk.requires_confirmation is False
