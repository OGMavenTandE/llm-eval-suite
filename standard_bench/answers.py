"""Answer text graded at run time and again on rescore.

One path. A chat template that opens ``<think>`` inside the prompt leaves
the completion without an opening tag. A marker such as ``<|end_of_thought|>``
closes the block the same way the generator does. Code suites keep a
leading indent.
"""

from __future__ import annotations

from llm_eval.models.context import answer_after_think, scored_after_thinking

CODE_SUITES = frozenset({"humaneval_plus", "mbpp_plus"})


def graded_answer(
    raw: str,
    *,
    suite: str = "",
    think_marker: str | None = None,
    prompt_opened_think: bool = False,
) -> str:
    """Text the scorer grades. ``runner`` and ``rescore`` both call this."""
    strip_edges = suite not in CODE_SUITES
    marker = (think_marker or "").strip()
    if marker and marker.lower() != "</think>":
        return scored_after_thinking(raw or "", marker, strip_edges=strip_edges)
    return answer_after_think(
        raw or "",
        prompt_opened_think=bool(prompt_opened_think),
        strip_edges=strip_edges,
    )
