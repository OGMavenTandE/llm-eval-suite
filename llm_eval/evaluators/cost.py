from .base import BaseEvaluator, EvalResult


class CostEvaluator(BaseEvaluator):
    def __init__(self, config: dict):
        super().__init__(config)
        self.cost_per_1k_tokens = config.get("cost_per_1k_tokens", 0.0)
        self.max_tokens_per_response = config.get("max_tokens_per_response", None)
        self.threshold = config.get("threshold", None)

    def evaluate(self, prompt: str, expected: str, response: str, **kwargs) -> EvalResult:
        tokens_used = kwargs.get("tokens_used")
        latency_ms = kwargs.get("latency_ms")

        if tokens_used is None:
            return EvalResult(
                score=1.0,
                passed=True,
                details={
                    "tokens_used": None,
                    "estimated_cost": None,
                    "cost_per_1k_tokens": self.cost_per_1k_tokens,
                    "tokens_per_response": None,
                    "note": "tokens not reported",
                },
                metric_name="cost",
            )

        if self.max_tokens_per_response is not None:
            score = 1.0 - min(tokens_used / self.max_tokens_per_response, 1.0)
        else:
            score = 1.0

        estimated_cost = (tokens_used / 1000) * self.cost_per_1k_tokens

        if self.threshold is not None:
            passed = estimated_cost <= self.threshold
        else:
            passed = True

        return EvalResult(
            score=score,
            passed=passed,
            details={
                "tokens_used": tokens_used,
                "estimated_cost": estimated_cost,
                "cost_per_1k_tokens": self.cost_per_1k_tokens,
                "tokens_per_response": tokens_used,
            },
            metric_name="cost",
        )

    @classmethod
    def summarize(cls, results: list[EvalResult]) -> dict:
        reported = [r for r in results if r.details.get("tokens_used") is not None]

        total_tokens = sum(r.details["tokens_used"] for r in reported)
        total_cost = sum(r.details["estimated_cost"] for r in reported)
        token_counts = [r.details["tokens_used"] for r in reported]

        return {
            "total_tokens": total_tokens,
            "total_estimated_cost": total_cost,
            "mean_tokens_per_response": total_tokens / len(reported) if reported else None,
            "min_tokens": min(token_counts) if reported else None,
            "max_tokens": max(token_counts) if reported else None,
        }
