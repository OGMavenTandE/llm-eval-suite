import random
import string

from .base import BaseEvaluator, EvalResult


_SYNONYM_MAP = {
    "good": "fine",
    "bad": "poor",
    "big": "large",
    "small": "tiny",
    "fast": "quick",
    "slow": "sluggish",
    "happy": "glad",
    "sad": "unhappy",
    "smart": "clever",
    "hard": "difficult",
    "easy": "simple",
    "show": "display",
    "tell": "explain",
    "get": "obtain",
    "make": "create",
    "use": "utilize",
}


def _perturb_typo(text: str) -> str:
    words = text.split()
    if not words:
        return text
    n_perturbations = random.randint(2, 3)
    for _ in range(n_perturbations):
        idx = random.randrange(len(words))
        word = words[idx]
        if len(word) < 2:
            continue
        op = random.choice(["insert", "delete", "swap"])
        chars = list(word)
        if op == "insert":
            pos = random.randrange(len(chars) + 1)
            chars.insert(pos, random.choice(string.ascii_lowercase))
        elif op == "delete":
            pos = random.randrange(len(chars))
            chars.pop(pos)
        elif op == "swap" and len(chars) >= 2:
            pos = random.randrange(len(chars) - 1)
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        words[idx] = "".join(chars)
    return " ".join(words)


def _perturb_case(text: str) -> str:
    words = text.split()
    result = []
    for word in words:
        choice = random.choice(["upper", "lower", "title", "keep"])
        if choice == "upper":
            result.append(word.upper())
        elif choice == "lower":
            result.append(word.lower())
        elif choice == "title":
            result.append(word.capitalize())
        else:
            result.append(word)
    return " ".join(result)


def _perturb_rephrase(text: str) -> str:
    words = text.split()
    # Synonym substitution
    words = [_SYNONYM_MAP.get(w.lower(), w) for w in words]
    # Simple word reorder: shuffle pairs of adjacent words
    for i in range(0, len(words) - 1, 2):
        if random.random() < 0.3:
            words[i], words[i + 1] = words[i + 1], words[i]
    return " ".join(words)


_PERTURBERS = {
    "typo": _perturb_typo,
    "case": _perturb_case,
    "rephrase": _perturb_rephrase,
}


class RobustnessEvaluator(BaseEvaluator):
    def __init__(self, config: dict):
        super().__init__(config)
        requested = config.get("perturbations", list(_PERTURBERS.keys()))
        self.perturbations = [p for p in requested if p in _PERTURBERS]

    def evaluate(self, prompt: str, expected: str, response: str, **kwargs) -> EvalResult:
        model = kwargs["model"]
        correctness_evaluator = kwargs["correctness_evaluator"]

        # Baseline correctness score
        baseline_result = correctness_evaluator.evaluate(prompt, expected, response)
        baseline_score = baseline_result.score if baseline_result.score > 0 else 1e-9

        perturbed_scores = []
        perturbation_details = {}

        for ptype in self.perturbations:
            perturber = _PERTURBERS[ptype]
            perturbed_prompt = perturber(prompt)
            model_response = model.generate(perturbed_prompt)
            eval_result = correctness_evaluator.evaluate(
                perturbed_prompt, expected, model_response.text
            )
            perturbed_scores.append(eval_result.score)
            perturbation_details[ptype] = {
                "perturbed_prompt": perturbed_prompt,
                "score": eval_result.score,
            }

        mean_perturbed = sum(perturbed_scores) / len(perturbed_scores) if perturbed_scores else 0.0
        consistency_score = min(mean_perturbed / baseline_score, 1.0)

        return EvalResult(
            score=consistency_score,
            passed=consistency_score >= self.config.get("threshold", 0.8),
            details={
                "baseline_score": baseline_score,
                "mean_perturbed_score": mean_perturbed,
                "perturbations": perturbation_details,
            },
            metric_name="robustness",
        )
