from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class EvalResult:
    score: float
    passed: bool
    details: dict
    metric_name: str


class BaseEvaluator(ABC):
    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def evaluate(self, prompt: str, expected: str, response: str, **kwargs) -> EvalResult:
        """
        Evaluate a model response against the expected output.

        Args:
            prompt:   The original input prompt sent to the model.
            expected: The ground-truth / reference answer.
            response: The model's generated response.
            **kwargs: Evaluator-specific extras (e.g. judge_model, latency_ms).

        Returns:
            EvalResult with score in [0, 1], pass/fail, details dict, and metric name.
        """
