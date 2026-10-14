"""Robustness perturbations. No model call."""

import random

from llm_eval.evaluators.robustness import _perturb_typo


def test_typo_perturbation_leaves_short_words_unchanged():
    """Words shorter than two characters are skipped. Longer words can change."""
    short = "a I x"
    for seed in range(20):
        random.seed(seed)
        assert _perturb_typo(short) == short

    changed = False
    for seed in range(30):
        random.seed(seed)
        words = _perturb_typo("I am").split()
        assert words[0] == "I"
        assert len(words) == 2
        if words[1] != "am":
            changed = True
    assert changed
