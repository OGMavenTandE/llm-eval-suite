"""Prompt budget helper for short-context models such as GPT-2 (1,024 tokens)."""


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
