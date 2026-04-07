import difflib

from .base import BaseEvaluator, EvalResult


class CorrectnessEvaluator(BaseEvaluator):
    def __init__(self, config: dict):
        super().__init__(config)
        self.mode = config.get("mode", "exact_match")
        self.threshold = config.get("threshold", 0.8)

    def evaluate(self, prompt: str, expected: str, response: str, **kwargs) -> EvalResult:
        match self.mode:
            case "exact_match":
                score = self._exact_match(expected, response)
            case "fuzzy_match":
                score = self._fuzzy_match(expected, response)
            case "llm_judge":
                score = self._llm_judge(prompt, expected, response, kwargs["judge_model"])
            case _:
                raise ValueError(f"Unknown correctness mode: {self.mode!r}")

        return EvalResult(
            score=score,
            passed=score >= self.threshold,
            details={"mode": self.mode, "expected": expected, "response": response},
            metric_name=f"correctness_{self.mode}",
        )

    def _exact_match(self, expected: str, response: str) -> float:
        return 1.0 if expected.strip().lower() == response.strip().lower() else 0.0

    def _fuzzy_match(self, expected: str, response: str) -> float:
        return difflib.SequenceMatcher(None, expected.strip(), response.strip()).ratio()

    def _llm_judge(self, prompt: str, expected: str, response: str, judge_model) -> float:
        judge_prompt = (
            "Rate the following response on a scale of 1-5 for correctness.\n"
            f"Question: {prompt}\n"
            f"Expected: {expected}\n"
            f"Response: {response}\n"
            "Score (1-5):"
        )
        result = judge_model.generate(judge_prompt)
        raw = result.text.strip()
        # Extract the first numeric token from the judge's reply
        for token in raw.split():
            token = token.strip(".,;:()")
            try:
                numeric = float(token)
                # Normalise 1-5 → 0-1
                score = (numeric - 1) / 4.0
                return max(0.0, min(1.0, score))
            except ValueError:
                continue
        # Fallback: could not parse — treat as zero
        return 0.0
