"""Native public capability benchmarks.

These suites are not part of the AIEval Department of War bench score.
"""

SUITE_VERSION = "standard-1"

SUITE_NAMES = (
    "mmlu_pro",
    "gpqa_diamond",
    "gsm8k",
    "math500",
    "ifeval",
    "humaneval_plus",
    "mbpp_plus",
)

# Public benchmarks never enter the Department of War composite.
COUNTS_TOWARD_DOW_SCORE = False
