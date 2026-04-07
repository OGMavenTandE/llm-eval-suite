from .base import BaseEvaluator, EvalResult
from .correctness import CorrectnessEvaluator
from .latency import LatencyEvaluator
from .robustness import RobustnessEvaluator
from .consistency import ConsistencyEvaluator
from .cost import CostEvaluator

__all__ = [
    "BaseEvaluator",
    "EvalResult",
    "CorrectnessEvaluator",
    "LatencyEvaluator",
    "RobustnessEvaluator",
    "ConsistencyEvaluator",
    "CostEvaluator",
    "EVALUATOR_REGISTRY",
]

# Maps YAML evaluator names → evaluator classes.
EVALUATOR_REGISTRY: dict[str, type[BaseEvaluator]] = {
    "correctness": CorrectnessEvaluator,
    "latency": LatencyEvaluator,
    "robustness": RobustnessEvaluator,
    "consistency": ConsistencyEvaluator,
    "cost": CostEvaluator,
}
