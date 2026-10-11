"""Customer subsets and the in-house GPQA opt-in.

``standard_capability`` and ``standard_capability_quick`` score six suites.
GPQA Diamond stays in the code. It is not part of those presets or the
customer zip.
"""

from __future__ import annotations

from standard_bench.catalog import SUITES

CUSTOMER_SUITES = (
    "mmlu_pro",
    "gsm8k",
    "math500",
    "ifeval",
    "humaneval_plus",
    "mbpp_plus",
)

GPQA_SUITE = "gpqa_diamond"

QUICK_SUBSET = "quick50"
QUICK_SEED = 20261011
QUICK_LIMIT = 50
QUICK_THINKING_BUDGET = 512

KNOWN_SUBSETS = ("full", "short", "quick50")

PROTOCOL_KEYS = ("subset", "seed", "thinking_budget", "protocol")

# First-run screen for the six customer suites. HumanEval+ is already small.
CUSTOMER_SHORT_LIMITS = {
    "mmlu_pro": 200,
    "gsm8k": 200,
    "math500": 100,
    "ifeval": 100,
    "humaneval_plus": 164,
    "mbpp_plus": 50,
}


def subset_limits(name: str) -> dict[str, int] | None:
    """Item caps for a named subset. ``full`` means no cap."""
    key = (name or "full").strip()
    if key == "full":
        return None
    if key == "quick50":
        return {suite: min(QUICK_LIMIT, spec.full_size) for suite, spec in SUITES.items()}
    if key == "short":
        return dict(CUSTOMER_SHORT_LIMITS)
    known = ", ".join(KNOWN_SUBSETS)
    raise KeyError(f"Unknown subset '{name}'. Known subsets: {known}.")


def protocol_name(subset: str) -> str:
    if subset == "quick50":
        return "quick-check"
    if subset == "short":
        return "short"
    return "full"
