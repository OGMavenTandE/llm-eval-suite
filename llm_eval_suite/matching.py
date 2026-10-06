"""Word-boundary matching for keyword and fact-check scores.

A capitalized word that continues a name is a partial match, not a pass.
Expected ``Newport`` does not pass on ``Newport News``.
"""

from __future__ import annotations

import re

WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*")


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
    needles = [word.casefold() for word, _start, _end in expected_words]
    width = len(needles)
    full = None
    partial = None
    for index in range(0, len(response_words) - width + 1):
        window = [response_words[index + offset][0].casefold() for offset in range(width)]
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
