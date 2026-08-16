from wordlehands.agent.solver import (
    consistent_with_guess,
    constraints_summary,
    filter_candidates,
    load_wordlist,
    propose,
    rank_candidates,
)


def test_consistent_with_guess_handles_duplicate_letters_correctly():
    # secret ROBOT has two O's; guess ROOMS has two O's too. The second O in
    # the guess (position 2) should read "elsewhere" (one O remains after the
    # green at position 1 consumes one), not "absent" and not "correct".
    assert consistent_with_guess(
        "robot", "rooms", ["correct", "correct", "elsewhere", "absent", "absent"]
    )
    assert not consistent_with_guess(
        "robot", "rooms", ["correct", "correct", "absent", "absent", "absent"]
    )


def test_consistent_with_guess_all_correct_only_for_exact_match():
    assert consistent_with_guess("crane", "crane", ["correct"] * 5)
    assert not consistent_with_guess("crane", "crane", ["correct", "correct", "correct", "correct", "elsewhere"])


def test_filter_candidates_narrows_pool_using_real_feedback():
    words = ["crane", "trace", "react", "cater", "irate", "which"]
    # ground truth: scoring guess "crane" against secret "irate" yields exactly this
    history = [("crane", ["absent", "correct", "correct", "absent", "correct"])]
    survivors = filter_candidates(words, history)
    assert survivors == ["irate"]


def test_rank_candidates_returns_bounded_list():
    ranked = rank_candidates(["crane", "trace", "react"], limit=2)
    assert len(ranked) == 2


def test_constraints_summary_separates_locked_present_and_excluded():
    history = [("crane", ["correct", "absent", "absent", "elsewhere", "absent"])]
    summary = constraints_summary(history)
    assert summary["locked_positions"] == {"0": "c"}
    assert "n" in summary["must_include"]
    assert "r" in summary["excluded"] and "a" in summary["excluded"] and "e" in summary["excluded"]


def test_propose_shrinks_as_history_grows():
    words = load_wordlist()
    first = propose(words, [])
    narrowed = propose(words, [("crane", ["absent", "absent", "absent", "correct", "correct"])])
    assert narrowed["remaining_possible_count"] < first["remaining_possible_count"]
    assert all(len(w) == 5 for w in narrowed["candidates"])


def test_wordlist_loads_and_is_reasonably_sized():
    words = load_wordlist()
    assert len(words) > 1000
    assert all(len(w) == 5 and w.isalpha() for w in words[:50])
