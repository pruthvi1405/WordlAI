"""ReAct-style next-word solver.

This is not the discovery LLM — it's a small deterministic helper the
discovery LLM calls as a tool (`propose_next_words` in agent/tools.py). Given
every (guess, feedback) pair submitted so far this game, it filters the
bundled dictionary down to exactly the words still consistent with all of it,
then ranks the survivors by letter-frequency coverage so the most
information-rich guess comes first.

This is the "Reason" half of the ReAct pattern: the constraint bookkeeping a
human Wordle player does by hand (which letters are locked, which are known
present-but-misplaced, which are ruled out) is exact and mechanical, so it
shouldn't be re-derived from a screenshot by the LLM every turn — that's both
wasteful and a chance to hallucinate a word that contradicts known feedback.
The LLM still does the "Act": it picks which candidate to actually submit.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

Feedback = str  # "correct" | "elsewhere" | "absent"

_WORDLIST_PATH = Path(__file__).parent / "data" / "words5.txt"


def load_wordlist() -> list[str]:
    return [w.strip() for w in _WORDLIST_PATH.read_text().splitlines() if w.strip()]


def consistent_with_guess(candidate: str, guess: str, feedback: list[Feedback]) -> bool:
    """Would submitting `guess` against secret word `candidate` have produced
    exactly `feedback`? Implemented by simulating the real scoring rule
    (greens consume letter counts first, then yellows/grays split what's
    left) rather than reasoning about `candidate` directly — this is the one
    formulation that's correct in the presence of repeated letters (e.g. a
    guess with two E's against a secret with only one) without hand-rolled
    special cases.
    """
    if len(candidate) != 5 or len(guess) != 5 or len(feedback) != 5:
        return False

    remaining = Counter(candidate)
    simulated: list[str] = [""] * 5

    for i in range(5):
        if guess[i] == candidate[i]:
            simulated[i] = "correct"
            remaining[guess[i]] -= 1

    for i in range(5):
        if simulated[i]:
            continue
        letter = guess[i]
        if remaining[letter] > 0:
            simulated[i] = "elsewhere"
            remaining[letter] -= 1
        else:
            simulated[i] = "absent"

    return simulated == list(feedback)


def filter_candidates(words: list[str], history: list[tuple[str, list[Feedback]]]) -> list[str]:
    survivors = words
    for guess, feedback in history:
        survivors = [w for w in survivors if consistent_with_guess(w, guess, feedback)]
    return survivors


def rank_candidates(candidates: list[str], limit: int = 15) -> list[str]:
    """Score each candidate by how common its unique letters are across the
    remaining candidate pool — a cheap information-gain proxy. A word built
    from letters that show up in many other survivors is more likely to
    split the remaining pool efficiently on the next guess.
    """
    if not candidates:
        return []
    letter_freq: Counter[str] = Counter()
    for w in candidates:
        for ch in set(w):
            letter_freq[ch] += 1

    def score(word: str) -> int:
        return sum(letter_freq[ch] for ch in set(word))

    return sorted(candidates, key=score, reverse=True)[:limit]


def constraints_summary(history: list[tuple[str, list[Feedback]]]) -> dict:
    """Human/LLM-readable summary of what's known so far. Informational only
    — filtering itself always goes through consistent_with_guess, which is
    exact; this is just for the tool result so the model doesn't have to
    reverse-engineer the constraints from the candidate list.
    """
    locked: dict[int, str] = {}
    present: set[str] = set()
    ruled_out: set[str] = set()
    for guess, feedback in history:
        for i, (letter, state) in enumerate(zip(guess, feedback)):
            if state == "correct":
                locked[i] = letter
                present.add(letter)
            elif state == "elsewhere":
                present.add(letter)
            elif state == "absent":
                ruled_out.add(letter)
    return {
        "locked_positions": {str(k): v for k, v in locked.items()},
        "must_include": sorted(present),
        "excluded": sorted(ruled_out - present),
    }


def propose(words: list[str], history: list[tuple[str, list[Feedback]]], limit: int = 15) -> dict:
    candidates = filter_candidates(words, history)
    return {
        "candidates": rank_candidates(candidates, limit=limit),
        "remaining_possible_count": len(candidates),
        "constraints": constraints_summary(history),
    }
