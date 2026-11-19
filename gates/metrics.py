"""Classification metrics with intervals. Pure Python, no numpy."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass
class ClassMetrics:
    label: str
    support: int
    predicted: int
    true_positive: int

    @property
    def recall(self) -> float | None:
        return self.true_positive / self.support if self.support else None

    @property
    def precision(self) -> float | None:
        return self.true_positive / self.predicted if self.predicted else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        if not p or not r:
            return 0.0 if (p is not None and r is not None) else None
        return 2 * p * r / (p + r)


@dataclass
class RunMetrics:
    name: str
    total: int
    parsed: int
    correct: int
    classes: dict[str, ClassMetrics]
    confusion: dict[str, dict[str, int]]  # truth -> prediction (or "UNPARSED") -> count
    hits: dict[str, bool] = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def parsed_rate(self) -> float:
        return self.parsed / self.total if self.total else 0.0

    @property
    def accuracy_interval(self) -> tuple[float, float]:
        return wilson_interval(self.correct, self.total)

    @property
    def macro_f1(self) -> float:
        scores = [c.f1 for c in self.classes.values() if c.support and c.f1 is not None]
        return sum(scores) / len(scores) if scores else 0.0

    def to_dict(self) -> dict:
        low, high = self.accuracy_interval
        return {
            "name": self.name,
            "total": self.total,
            "parsed": self.parsed,
            "parsed_rate": self.parsed_rate,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "accuracy_ci95": [low, high],
            "macro_f1": self.macro_f1,
            "classes": {
                label: {
                    "support": c.support,
                    "predicted": c.predicted,
                    "true_positive": c.true_positive,
                    "recall": c.recall,
                    "precision": c.precision,
                    "f1": c.f1,
                }
                for label, c in self.classes.items()
            },
            "confusion": self.confusion,
        }


UNPARSED = "UNPARSED"


def score_run(name: str, items: list[dict], predictions: dict[str, dict], labels: list[str]) -> RunMetrics:
    """Score predictions against item labels. A missing or unparsed prediction is wrong."""
    classes = {label: ClassMetrics(label, 0, 0, 0) for label in labels}
    confusion = {label: {} for label in labels}
    parsed = correct = 0
    hits: dict[str, bool] = {}
    for item in items:
        truth = item["label"]
        if truth not in classes:
            raise ValueError(f"item {item['id']!r} has label {truth!r}, which is not in the spec labels")
        pred = (predictions.get(item["id"]) or {}).get("prediction")
        classes[truth].support += 1
        key = pred if pred is not None else UNPARSED
        confusion[truth][key] = confusion[truth].get(key, 0) + 1
        if pred is not None:
            parsed += 1
            classes[pred].predicted += 1
        hit = pred == truth
        hits[item["id"]] = hit
        if hit:
            correct += 1
            classes[truth].true_positive += 1
    return RunMetrics(name, len(items), parsed, correct, classes, confusion, hits)


def paired_bootstrap_delta(
    candidate: dict[str, bool],
    baseline: dict[str, bool],
    *,
    resamples: int = 2000,
    seed: int = 0,
) -> dict:
    """Accuracy difference (candidate minus baseline) on the same items, with a 95% interval."""
    ids = sorted(set(candidate) & set(baseline))
    if not ids:
        return {"items": 0, "delta": 0.0, "ci95": [0.0, 0.0]}
    diffs = [int(candidate[i]) - int(baseline[i]) for i in ids]
    n = len(diffs)
    rng = random.Random(seed)
    means = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    low = means[int(0.025 * (resamples - 1))]
    high = means[int(math.ceil(0.975 * (resamples - 1)))]
    both_right = sum(1 for i in ids if candidate[i] and baseline[i])
    both_wrong = sum(1 for i in ids if not candidate[i] and not baseline[i])
    return {
        "items": n,
        "delta": sum(diffs) / n,
        "ci95": [low, high],
        "both_right": both_right,
        "both_wrong": both_wrong,
        "candidate_only_right": sum(1 for d in diffs if d == 1),
        "baseline_only_right": sum(1 for d in diffs if d == -1),
    }
