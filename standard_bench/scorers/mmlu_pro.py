"""MMLU-Pro 10-option chain-of-thought scoring.

The prompt and the three extractors follow TIGER-AI-Lab/MMLU-Pro
``evaluate_from_local.py``. A failed extraction is incorrect. The official
script would draw a random letter. This runner does not.
"""

from __future__ import annotations

import re

LETTERS = "ABCDEFGHIJKLMNOP"

_ANSWER_IS = re.compile(r"answer is \(?([A-J])\)?")
_ANSWER_COLON = re.compile(r".*[aA]nswer:\s*([A-J])", re.DOTALL)
_LAST_LETTER = re.compile(r"\b[A-J]\b(?!.*\b[A-J]\b)", re.DOTALL)

INITIAL_PROMPT = (
    'The following are multiple choice questions (with answers) about {category}. '
    'Think step by step and then finish your answer with "the answer is (X)" '
    "where X is the correct letter choice.\n\n"
)


def extract_final(text: str) -> str | None:
    match = _LAST_LETTER.search(text or "")
    if not match:
        return None
    return match.group(0)


def extract_again(text: str) -> str | None:
    match = _ANSWER_COLON.search(text or "")
    if match:
        return match.group(1)
    return extract_final(text)


def extract_answer(text: str) -> str | None:
    """Official three-step extractor. No random guess on failure."""
    match = _ANSWER_IS.search(text or "")
    if match:
        return match.group(1)
    return extract_again(text)


def format_options(options: list[str]) -> str:
    lines = ["Options:"]
    for index, option in enumerate(options):
        lines.append(f"{LETTERS[index]}. {option}")
    return "\n".join(lines)


def format_example(row: dict, *, including_answer: bool) -> str:
    question = str(row.get("question") or "")
    options = list(row.get("options") or [])
    prompt = "Question:\n" + question + "\n" + format_options(options) + "\n"
    if including_answer:
        cot = str(row.get("cot_content") or "")
        cot = cot.replace(
            "A: Let's think step by step.",
            "Answer: Let's think step by step.",
        )
        prompt += cot + "\n\n"
    else:
        prompt += "Answer: Let's think step by step."
    return prompt


def build_prompt(row: dict, shots: list[dict]) -> str:
    """5-shot CoT for the question's category. ``shots`` are already filtered."""
    category = str(row.get("category") or "the subject")
    prompt = INITIAL_PROMPT.format(category=category)
    for shot in shots[:5]:
        prompt += format_example(shot, including_answer=True)
    prompt += format_example(row, including_answer=False)
    return prompt


def score_mmlu_pro(gold: str, response: str) -> dict:
    predicted = extract_answer(response)
    expected = (gold or "").strip().upper()[:1]
    passed = bool(predicted) and predicted == expected
    return {
        "predicted": predicted or "",
        "expected": expected,
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "metric": "accuracy",
    }
