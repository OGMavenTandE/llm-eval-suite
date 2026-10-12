"""Prompt budget helper for short-context models such as GPT-2 (1,024 tokens)."""

from __future__ import annotations

import re

_THINK_CLOSED = re.compile(r"<think\b[^>]*>.*?</think>", re.IGNORECASE | re.DOTALL)
_THINK_OPEN = re.compile(r"<think\b[^>]*>.*", re.IGNORECASE | re.DOTALL)


def strip_think_blocks(text: str | None, *, strip_edges: bool = True) -> str:
    """Drop model thinking traces so the scored text is the answer.

    A closed ``<think>`` block is removed. An unclosed block, which is what
    remains when thinking tokens use up ``max_tokens``, is removed through
    the end of the string. Code completions pass ``strip_edges=False`` so a
    leading indent on the function body is kept.
    """
    if not text:
        return ""
    cleaned = _THINK_CLOSED.sub("", text)
    cleaned = _THINK_OPEN.sub("", cleaned)
    if strip_edges:
        return cleaned.strip()
    return cleaned


_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)
_THINK_OPEN_TAG = re.compile(r"<think\b", re.IGNORECASE)


def prompt_opened_think(prompt: str) -> bool:
    """True when the chat template left an unclosed ``<think>`` on the prompt."""
    tail = (prompt or "")[-120:].lower()
    return "<think>" in tail and "</think>" not in tail


def answer_after_think(
    raw: str,
    *,
    prompt_opened_think: bool = False,
    strip_edges: bool = True,
) -> str:
    """Scored text. Drops a closed think block and an unterminated one.

    When the template opened ``<think>`` in the prompt, the completion
    starts inside that block. Text through ``</think>`` is dropped. If
    the close tag never arrives, the completion is still thinking.
    ``strip_edges`` is false for code so a leading indent stays.
    """
    text = raw or ""
    if prompt_opened_think and not _THINK_OPEN_TAG.search(text):
        close = _THINK_CLOSE.search(text)
        if close is None:
            return ""
        text = text[close.end() :]
    return strip_think_blocks(text, strip_edges=strip_edges)


def think_split(text: str, marker: str) -> tuple[bool, str]:
    """Whether ``marker`` closed the think block, and the text after it."""
    if not text or not marker:
        return False, ""
    index = text.lower().find(marker.lower())
    if index < 0:
        return False, ""
    return True, text[index + len(marker) :]


def scored_after_thinking(raw_text: str, marker: str, *, strip_edges: bool = True) -> str:
    """Answer text after the think block. The block itself is not scored.

    ``</think>`` uses the template-opened path, because a reasoning
    completion starts inside the block the chat template opened. Any other
    marker, including ``<|end_of_thought|>``, is the close tag itself.
    """
    if (marker or "").lower() == "</think>":
        return answer_after_think(raw_text, prompt_opened_think=True, strip_edges=strip_edges)
    closed, after = think_split(raw_text, marker)
    if closed:
        return strip_think_blocks(after, strip_edges=strip_edges)
    return ""


def clamp_prompt_and_new_tokens(prompt: str, max_new_tokens: int, max_context: int) -> tuple[str, int]:
    """Keep ``prompt + max_new_tokens`` inside ``max_context``.

    The prompt is shortened from the left (the tail is kept) and new tokens
    are capped at ``max_context - 1``. Whitespace-separated words stand in
    for tokens when a tokenizer is not loaded.
    """
    context = int(max_context)
    if context <= 1:
        return "", 1
    max_new = max(1, min(int(max_new_tokens), context - 1))
    budget = max(1, context - max_new)
    return truncate_to_token_budget(prompt or "", budget), max_new


def truncate_to_token_budget(text: str, budget: int) -> str:
    """Keep the tail of ``text`` so a short context window still sees the question.

    Whitespace-separated words stand in for tokens when a tokenizer is not loaded.
    """
    if budget <= 0:
        return ""
    words = text.split()
    if len(words) <= budget:
        return text
    return " ".join(words[-budget:])
