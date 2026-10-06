"""Word-boundary matching for keyword and fact-check scores.

A capitalized word that continues a name is a partial match, not a pass.
Expected ``Newport`` does not pass on ``Newport News``.
"""

from __future__ import annotations

import re

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")

# Zero through twenty, plus the tens. Used so "Eight" matches an expected "8".
_NUMBER_WORDS = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "thirteen": "13",
    "fourteen": "14",
    "fifteen": "15",
    "sixteen": "16",
    "seventeen": "17",
    "eighteen": "18",
    "nineteen": "19",
    "twenty": "20",
    "thirty": "30",
    "forty": "40",
    "fifty": "50",
    "sixty": "60",
    "seventy": "70",
    "eighty": "80",
    "ninety": "90",
}


def _canon_word(word: str) -> str:
    folded = word.casefold()
    return _NUMBER_WORDS.get(folded, folded)


def _words(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(0), match.start(), match.end()) for match in WORD_RE.finditer(text or "")]


def _excerpt(text: str, start: int, end: int, radius: int = 48) -> str:
    if not text:
        return ""
    left = max(0, start - radius)
    right = min(len(text), end + radius)
    snippet = text[left:right].replace("\n", " ")
    if left > 0:
        snippet = "..." + snippet
    if right < len(text):
        snippet = snippet + "..."
    return snippet


def match_expected(expected: str, response: str) -> dict:
    """Return the best word-sequence match and whether it is a full pass."""
    expected_words = _words(expected or "")
    response_words = _words(response or "")
    excerpt = _excerpt(response or "", 0, min(len(response or ""), 80))
    if not expected_words:
        return {"kind": "none", "passed": False, "span": "", "excerpt": excerpt, "start": 0, "end": 0}
    needles = [_canon_word(word) for word, _start, _end in expected_words]
    width = len(needles)
    full = None
    partial = None
    for index in range(0, len(response_words) - width + 1):
        window = [_canon_word(response_words[index + offset][0]) for offset in range(width)]
        if window != needles:
            continue
        start = response_words[index][1]
        end = response_words[index + width - 1][2]
        continued = False
        if index + width < len(response_words):
            next_word, next_start, next_end = response_words[index + width]
            gap = (response or "")[end:next_start]
            last_expected = expected_words[-1][0]
            if gap.strip() == "" and next_word[:1].isupper() and last_expected[:1].isupper():
                continued = True
                end = next_end
        span = (response or "")[start:end]
        found = {
            "kind": "partial" if continued else "full",
            "passed": not continued,
            "span": span,
            "excerpt": _excerpt(response or "", start, end),
            "start": start,
            "end": end,
        }
        if continued and partial is None:
            partial = found
        if not continued:
            full = found
            break
    if full is not None:
        return full
    if partial is not None:
        return partial
    return {"kind": "none", "passed": False, "span": "", "excerpt": excerpt, "start": 0, "end": 0}
