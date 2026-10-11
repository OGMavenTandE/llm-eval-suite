"""MATH-500 boxed answers with a sympy-free normalizer.

Numeric values and simple fractions match. Symbolic rearrangement does not.
``x+1`` and ``1+x`` are not equal. That limit is intentional and documented.
"""

from __future__ import annotations

import re

_FRAC = re.compile(r"\\(?:d|t)?frac\{([^{}]+)\}\{([^{}]+)\}")
_TEXT = re.compile(r"\\text\{([^{}]*)\}")
_BOXED = "\\boxed"
_RATIO = re.compile(
    r"^\(?(-?\d+(?:\.\d+)?)\)?/\(?(-?\d+(?:\.\d+)?)\)?$"
)

PROMPT = (
    "Solve the following math problem. Put the answer after the step-by-step "
    "solution, not before. Please put your final answer within \\boxed{{}}.\n\n"
    "{problem}"
)

_TOLERANCE = 1e-6


def build_prompt(problem: str) -> str:
    return PROMPT.format(problem=problem)


def extract_boxed(text: str) -> str | None:
    """Last ``\\boxed{...}`` with nested braces. A missing box is a miss."""
    source = text or ""
    start = source.rfind(_BOXED)
    if start < 0:
        return None
    brace = source.find("{", start)
    if brace < 0:
        return None
    depth = 0
    for index in range(brace, len(source)):
        char = source[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[brace + 1 : index]
    return None


def _expand_frac(text: str) -> str:
    current = text
    for _ in range(8):
        updated = _FRAC.sub(r"(\1)/(\2)", current)
        if updated == current:
            return current
        current = updated
    return current


def normalize_expr(expr: str) -> str:
    text = str(expr or "").strip()
    text = text.replace("$", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace("\\!", "").replace("\\;", "")
    text = text.replace("\\%", "%")
    text = _TEXT.sub(r"\1", text)
    text = _expand_frac(text)
    text = text.replace(" ", "").replace("\n", "")
    text = text.replace("\\(", "").replace("\\)", "")
    text = text.replace("\\[", "").replace("\\]", "")
    if text.endswith("."):
        text = text[:-1]
    return text


def _as_float(text: str) -> float | None:
    if re.fullmatch(r"-?\d+(?:\.\d+)?", text):
        return float(text)
    match = _RATIO.fullmatch(text)
    if not match:
        return None
    denominator = float(match.group(2))
    if denominator == 0:
        return None
    return float(match.group(1)) / denominator


def math_equal(left: str, right: str) -> bool:
    a = normalize_expr(left)
    b = normalize_expr(right)
    if a == b:
        return True
    # A trailing percent is part of the value. 50% does not equal 50.
    a_pct = a.endswith("%")
    b_pct = b.endswith("%")
    if a_pct != b_pct:
        return False
    if a_pct:
        a = a[:-1]
        b = b[:-1]
    va = _as_float(a)
    vb = _as_float(b)
    if va is None or vb is None:
        return False
    scale = max(1.0, abs(va), abs(vb))
    return abs(va - vb) <= _TOLERANCE * scale


def score_math(gold: str, response: str) -> dict:
    predicted = extract_boxed(response)
    passed = predicted is not None and math_equal(predicted, gold)
    return {
        "predicted": predicted if predicted is not None else "",
        "expected": gold,
        "passed": passed,
        "score": 1.0 if passed else 0.0,
        "metric": "accuracy",
    }
