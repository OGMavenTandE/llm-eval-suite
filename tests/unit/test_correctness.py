"""Correctness evaluator. No network."""

import pytest

from llm_eval.evaluators.correctness import CorrectnessEvaluator
from llm_eval.models.base import ModelResponse


class _Judge:
    def __init__(self, text: str):
        self.text = text

    def generate(self, prompt: str) -> ModelResponse:
        return ModelResponse(text=self.text, latency_ms=1.0, tokens_used=1)


def test_exact_match_ignores_case_and_surrounding_space():
    evaluator = CorrectnessEvaluator({"mode": "exact_match", "threshold": 0.8})
    passed = evaluator.evaluate("Capital?", "Paris", "  paris ")
    missed = evaluator.evaluate("Capital?", "Paris", "Lyon")
    assert passed.score == 1.0
    assert passed.passed is True
    assert passed.metric_name == "correctness_exact_match"
    assert missed.score == 0.0
    assert missed.passed is False


def test_fuzzy_match_scores_similarity():
    evaluator = CorrectnessEvaluator({"mode": "fuzzy_match", "threshold": 0.8})
    same = evaluator.evaluate("Capital?", "Paris", "Paris")
    close = evaluator.evaluate("Capital?", "Paris", "Par")
    assert same.score == 1.0
    assert same.passed is True
    assert 0.0 < close.score < 1.0
    assert close.metric_name == "correctness_fuzzy_match"


def test_llm_judge_reads_the_first_number_on_a_1_to_5_scale():
    evaluator = CorrectnessEvaluator({"mode": "llm_judge", "threshold": 0.8})
    top = evaluator.evaluate("Capital?", "Paris", "Paris", judge_model=_Judge("Score: 5"))
    mid = evaluator.evaluate("Capital?", "Paris", "Lyon", judge_model=_Judge("The score is 3."))
    low = evaluator.evaluate("Capital?", "Paris", "Lyon", judge_model=_Judge("1"))
    blank = evaluator.evaluate("Capital?", "Paris", "Lyon", judge_model=_Judge("no number here"))
    assert top.score == 1.0
    assert top.passed is True
    assert mid.score == 0.5
    assert mid.passed is False
    assert low.score == 0.0
    assert blank.score == 0.0
    assert blank.metric_name == "correctness_llm_judge"


def test_unknown_correctness_mode_raises():
    evaluator = CorrectnessEvaluator({"mode": "bleu"})
    with pytest.raises(ValueError, match="Unknown correctness mode: 'bleu'"):
        evaluator.evaluate("Capital?", "Paris", "Paris")
