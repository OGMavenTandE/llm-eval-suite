"""GPU-time planning numbers for one RTX 4070 Super (12 GB).

The repository does not record stage-1 tokens per second. The README's
4070 Super notes are gpt2-medium accuracy counts, not throughput. The rates
below are planning assumptions, stated as such.
"""

from __future__ import annotations

from standard_bench.catalog import SUITES

# prompt tokens, typical generated answer tokens. Thinking models add the
# 2048 thinking cap on top of the answer tokens. Prefill is charged at the
# decode rate, which overstates prefill and keeps the plan conservative.
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

SCREEN_LIMITS = {
    "mmlu_pro": 200,
    "gpqa_diamond": 198,
    "gsm8k": 200,
    "math500": 100,
    "ifeval": 100,
    "humaneval_plus": 164,
    "mbpp_plus": 50,
}


def tokens_per_item(suite: str, *, thinking: bool) -> int:
    budget = TOKEN_BUDGETS[suite]
    generated = budget["answer"] + (THINKING_CAP if thinking else 0)
    return budget["prompt"] + generated


def suite_seconds(suite: str, tok_s: float, *, thinking: bool, limit: int | None = None) -> float:
    count = SUITES[suite].full_size if limit is None else min(SUITES[suite].full_size, int(limit))
    return count * tokens_per_item(suite, thinking=thinking) / float(tok_s)


def model_seconds(model_key: str, limits: dict[str, int] | None = None) -> dict:
    tok_s = THROUGHPUT[model_key]
    thinking = model_key in THINKING_MODELS
    per_suite = {}
    total = 0.0
    for name in SUITES:
        limit = None if limits is None else limits.get(name)
        seconds = suite_seconds(name, tok_s, thinking=thinking, limit=limit)
        per_suite[name] = seconds
        total += seconds
    return {
        "model": model_key,
        "tok_s": tok_s,
        "thinking": thinking,
        "per_suite_seconds": per_suite,
        "total_seconds": total,
        "total_hours": total / 3600.0,
    }
