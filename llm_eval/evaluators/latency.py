import statistics

from .base import BaseEvaluator, EvalResult


class LatencyEvaluator(BaseEvaluator):
    def __init__(self, config: dict):
        super().__init__(config)
        self.max_ms = config.get("max_ms", 5000)

    def evaluate(self, prompt: str, expected: str, response: str, **kwargs) -> EvalResult:
        latency_ms = kwargs["latency_ms"]
        score = 1.0 - min(latency_ms / self.max_ms, 1.0)
        return EvalResult(
            score=score,
            passed=latency_ms <= self.max_ms,
            details={"latency_ms": latency_ms, "max_ms": self.max_ms},
            metric_name="latency",
        )

    @classmethod
    def summarize(cls, results: list[EvalResult]) -> dict:
        values = sorted(r.details["latency_ms"] for r in results)
        n = len(values)
        if n == 0:
            return {}

        def percentile(data, pct):
            idx = (pct / 100) * (len(data) - 1)
            lo, hi = int(idx), min(int(idx) + 1, len(data) - 1)
            frac = idx - lo
            return data[lo] + frac * (data[hi] - data[lo])

        return {
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "p95": percentile(values, 95),
            "p99": percentile(values, 99),
            "min": values[0],
            "max": values[-1],
        }
