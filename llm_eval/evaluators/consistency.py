import difflib
from itertools import combinations

from .base import BaseEvaluator, EvalResult


class ConsistencyEvaluator(BaseEvaluator):
    def __init__(self, config: dict):
        super().__init__(config)
        self.num_runs = config.get("num_runs", 5)
        self.threshold = config.get("threshold", 0.8)

    def evaluate(self, prompt: str, expected: str, response: str, **kwargs) -> EvalResult:
        model = kwargs["model"]

        responses = [model.generate(prompt).text for _ in range(self.num_runs)]

        unique_responses = list(dict.fromkeys(responses))

        if len(unique_responses) == 1:
            score = 1.0
            pairwise_scores = [1.0] * (self.num_runs * (self.num_runs - 1) // 2)
        else:
            pairs = list(combinations(responses, 2))
            pairwise_scores = [
                difflib.SequenceMatcher(None, a, b).ratio()
                for a, b in pairs
            ]
            score = sum(pairwise_scores) / len(pairwise_scores)

        return EvalResult(
            score=score,
            passed=score >= self.threshold,
            details={
                "responses": responses,
                "num_runs": self.num_runs,
                "pairwise_scores": pairwise_scores,
                "unique_response_count": len(unique_responses),
            },
            metric_name="consistency",
        )

    @classmethod
    def summarize(cls, results: list[EvalResult]) -> dict:
        scores = [r.score for r in results]
        return {
            "mean_consistency": sum(scores) / len(scores),
            "min_consistency": min(scores),
            "max_consistency": max(scores),
        }
