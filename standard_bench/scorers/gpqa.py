"""GPQA Diamond multiple choice with a seeded shuffle.

Question text stays in memory for the model call. Callers must not write it
to items, logs, or exports.
"""

from __future__ import annotations

import random
import re

SHUFFLE_SEED = 20261010

_ANSWER_LINE = re.compile(r"Answer:\s*([A-D])\b", re.IGNORECASE)
_LAST_LETTER = re.compile(r"\b([A-D])\b")

PROMPT = (
    "Answer the following multiple choice question. The last line of your response "
    "should be of the following format: 'Answer: $LETTER' (without quotes) where "
    "LETTER is one of ABCD. Think step by step before answering.\n\n"
    "{question}\n\n"
    "A) {a}\n"
    "B) {b}\n"
    "C) {c}\n"
    "D) {d}"
)


def shuffle_options(record_id: str, correct: str, incorrect: list[str]) -> tuple[list[str], str]:
    """Return four options and the correct letter. The shuffle is stable."""
    options = [correct, *list(incorrect)[:3]]
    while len(options) < 4:
        options.append("")
    rng = random.Random(f"{SHUFFLE_SEED}:{record_id}")
    order = list(range(4))
    rng.shuffle(order)
    shuffled = [options[index] for index in order]
    correct_index = order.index(0)
    return shuffled, "ABCD"[correct_index]


def build_prompt(question: str, options: list[str]) -> str:
    return PROMPT.format(
        question=question,
        a=options[0],
        b=options[1],
        c=options[2],
        d=options[3],
    )


def extract_answer(text: str) -> str | None:
    matches = _ANSWER_LINE.findall(text or "")
    if matches:
        return matches[-1].upper()
    letters = _LAST_LETTER.findall(text or "")
    if not letters:
        return None
    return letters[-1].upper()


def score_gpqa(gold_letter: str, response: str) -> dict:
    predicted = extract_answer(response)
    expected = (gold_letter or "").strip().upper()[:1]
    passed = bool(predicted) and predicted == expected
    return {
        "predicted": predicted or "",
        "expected": expected,
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "metric": "accuracy",
    }
