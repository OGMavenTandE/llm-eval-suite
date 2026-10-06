"""Prompt budget helper for short-context models such as GPT-2 (1,024 tokens)."""


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
