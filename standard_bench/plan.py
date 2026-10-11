"""GPU-time planning numbers for one RTX 4070 Super (12 GB).

The repository does not record stage-1 tokens per second. The README's
4070 Super notes are gpt2-medium accuracy counts, not throughput. The rates
below are planning assumptions, stated as such.
"""

from __future__ import annotations

from standard_bench.catalog import SUITES
from standard_bench.protocol import (
    CUSTOMER_SHORT_LIMITS,
    CUSTOMER_SUITES,
    QUICK_THINKING_BUDGET,
)

# prompt tokens, typical generated answer tokens. Thinking models add the
# 2048 thinking cap on top of the answer tokens. Prompt tokens are charged
# at PREFILL_TOK_S. Generated tokens are charged at the decode rate.
# Both rates are assumptions, not measurements.
TOKEN_BUDGETS = {
    "mmlu_pro": {"prompt": 2800, "answer": 350},
    "gpqa_diamond": {"prompt": 450, "answer": 400},
    "gsm8k": {"prompt": 900, "answer": 180},
    "math500": {"prompt": 250, "answer": 400},
    "ifeval": {"prompt": 180, "answer": 220},
    "humaneval_plus": {"prompt": 280, "answer": 180},
    "mbpp_plus": {"prompt": 220, "answer": 160},
}

THINKING_CAP = 2048

# Assumed prefill tokens per second on an RTX 4070 Super, 12 GB, batch 1.
# This is not a measurement from this repository.
PREFILL_TOK_S = 1500

# Assumed decode tokens per second on an RTX 4070 Super, 12 GB, batch 1.
THROUGHPUT = {
    "olmo-2-1b": 80,
    "lfm-1.2b": 80,
    "granite-1b": 80,
    "phi-4-mini-instruct": 45,
    "gemma-4-e2b-8bit": 40,
    "lfm-2.6b": 35,
    "granite-4.2-3b": 35,
    "phi-4-mini-reasoning": 35,
    "nemotron-3-nano-4b": 35,
}

THINKING_MODELS = {
    "lfm-2.6b",
    "granite-4.2-3b",
    "phi-4-mini-reasoning",
    "nemotron-3-nano-4b",
}

# Customer short subset. GPQA is not in this map.
SCREEN_LIMITS = dict(CUSTOMER_SHORT_LIMITS)

# Display order for the customer run plan. GPQA is not included.
PLAN_SUITES = CUSTOMER_SUITES


def tokens_per_item(suite: str, *, thinking: bool, thinking_tokens: int | None = None) -> int:
    budget = TOKEN_BUDGETS[suite]
    cap = 0
    if thinking:
        cap = THINKING_CAP if thinking_tokens is None else int(thinking_tokens)
    generated = budget["answer"] + cap
    return budget["prompt"] + generated


def suite_seconds(
    suite: str,
    tok_s: float,
    *,
    thinking: bool,
    limit: int | None = None,
    thinking_tokens: int | None = None,
) -> float:
    count = SUITES[suite].full_size if limit is None else min(SUITES[suite].full_size, int(limit))
    budget = TOKEN_BUDGETS[suite]
    cap = 0
    if thinking:
        cap = THINKING_CAP if thinking_tokens is None else int(thinking_tokens)
    generated = budget["answer"] + cap
    per_item = (budget["prompt"] / float(PREFILL_TOK_S)) + (generated / float(tok_s))
    return count * per_item


def model_seconds(
    model_key: str,
    limits: dict[str, int] | None = None,
    *,
    suites: tuple[str, ...] | None = None,
    thinking_tokens: int | None = None,
) -> dict:
    """Hours for the customer suites. GPQA is omitted unless ``suites`` lists it.

    ``thinking_tokens`` replaces the 2048 planning cap for thinking models.
    Non-thinking models still generate answer tokens only.
    """
    tok_s = THROUGHPUT[model_key]
    thinking = model_key in THINKING_MODELS
    names = tuple(suites) if suites is not None else PLAN_SUITES
    per_suite = {}
    total = 0.0
    for name in names:
        if limits is not None and name not in limits:
            continue
        limit = None if limits is None else limits.get(name)
        seconds = suite_seconds(
            name,
            tok_s,
            thinking=thinking,
            limit=limit,
            thinking_tokens=thinking_tokens,
        )
        per_suite[name] = seconds
        total += seconds
    applied = thinking_tokens if thinking and thinking_tokens is not None else (THINKING_CAP if thinking else 0)
    return {
        "model": model_key,
        "tok_s": tok_s,
        "thinking": thinking,
        "thinking_tokens": applied,
        "per_suite_seconds": per_suite,
        "total_seconds": total,
        "total_hours": total / 3600.0,
    }


def quick_limits(suites: tuple[str, ...] | None = None) -> dict[str, int]:
    names = tuple(suites) if suites is not None else PLAN_SUITES
    return {name: min(50, SUITES[name].full_size) for name in names}


# Re-export so the run plan and tests share one cap.
QUICK_THINKING_CAP = QUICK_THINKING_BUDGET
