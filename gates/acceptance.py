"""Gate 2 (baseline measurement) and Gate 5 (acceptance against the model it replaces)."""

from __future__ import annotations

from dataclasses import dataclass, field

from gates import GO, INVALID, NO_GO
from gates.metrics import RunMetrics, paired_bootstrap_delta, score_run
from gates.spec import GateSpec


@dataclass
class Check:
    name: str
    passed: bool
    detail: str
    blocks_validity: bool = False  # a failed validity check makes the gate INVALID, not NO-GO


@dataclass
class GateDecision:
    gate: int
    title: str
    workload: str
    verdict: str = GO
    checks: list[Check] = field(default_factory=list)
    metrics: dict = field(default_factory=dict)
    findings: list[str] = field(default_factory=list)
    inputs: dict = field(default_factory=dict)
    notice: str = ""  # shown as a banner, e.g. for simulated demo inputs

    def decide(self) -> "GateDecision":
        if any(not c.passed and c.blocks_validity for c in self.checks):
            self.verdict = INVALID
        elif any(not c.passed for c in self.checks):
            self.verdict = NO_GO
        else:
            self.verdict = GO
        return self

    def to_dict(self) -> dict:
        return {
            "gate": self.gate,
            "title": self.title,
            "workload": self.workload,
            "verdict": self.verdict,
            "checks": [c.__dict__ for c in self.checks],
            "metrics": self.metrics,
            "findings": self.findings,
            "inputs": self.inputs,
            "notice": self.notice,
        }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _validity_checks(spec: GateSpec, run: RunMetrics) -> list[Check]:
    acc = spec.acceptance
    return [
        Check(
            f"{run.name}: enough held-out items",
            run.total >= acc.min_items,
            f"{run.total} items scored; the spec requires at least {acc.min_items}.",
            blocks_validity=True,
        ),
        Check(
            f"{run.name}: replies parse to a label",
            run.parsed_rate >= acc.min_parsed,
            f"{run.parsed} of {run.total} replies parsed ({_pct(run.parsed_rate)}); "
            f"the spec requires {_pct(acc.min_parsed)}. Unparsed replies are scored wrong.",
            blocks_validity=True,
        ),
    ]


def _bar_checks(spec: GateSpec, run: RunMetrics) -> list[Check]:
    acc = spec.acceptance
    checks = [
        Check(
            f"{run.name}: accuracy at or above the mission bar",
            run.accuracy >= acc.min_accuracy,
            f"{run.correct} of {run.total} correct ({_pct(run.accuracy)}, 95% interval "
            f"{_pct(run.accuracy_interval[0])} to {_pct(run.accuracy_interval[1])}); "
            f"bar {_pct(acc.min_accuracy)}.",
        )
    ]
    for label, rule in spec.classes.items():
        if rule.min_recall is None:
            continue
        cls = run.classes[label]
        checks.append(
            Check(
                f"{run.name}: {label} recall at or above its bar",
                cls.recall is not None and cls.recall >= rule.min_recall,
                f"{cls.true_positive} of {cls.support} {label} items found ({_pct(cls.recall)}); "
                f"bar {_pct(rule.min_recall)}. Cost of a miss: {rule.cost_of_miss}.",
            )
        )
    return checks


def gate2_baseline(spec: GateSpec, items: list[dict], baseline_preds: dict[str, dict]) -> GateDecision:
    """Measure the model in use today. Its score is the bar Gate 5 compares against."""
    run = score_run(spec.baseline_name, items, baseline_preds, spec.labels)
    decision = GateDecision(2, "Independent quality measurement", spec.workload)
    decision.checks = _validity_checks(spec, run) + _bar_checks(spec, run)
    decision.metrics = {"baseline": run.to_dict()}
    decision.decide()
    if decision.verdict == NO_GO:
        decision.findings.append(
            "The model in use today does not meet the mission bar on this test set. Before specializing, "
            "check whether the bar, the labels, or the workflow needs to change."
        )
    elif decision.verdict == GO:
        decision.findings.append(
            f"Baseline accuracy {_pct(run.accuracy)} is the score a specialized model must match at Gate 5."
        )
    return decision


def gate5_acceptance(
    spec: GateSpec,
    items: list[dict],
    candidate_preds: dict[str, dict],
    baseline_preds: dict[str, dict],
    *,
    candidate_name: str = "candidate",
) -> GateDecision:
    """The candidate must pass the same tests as the model it would replace, on held-out items."""
    acc = spec.acceptance
    cand = score_run(candidate_name, items, candidate_preds, spec.labels)
    base = score_run(spec.baseline_name, items, baseline_preds, spec.labels)
    delta = paired_bootstrap_delta(cand.hits, base.hits, resamples=acc.bootstrap_resamples, seed=acc.seed)

    decision = GateDecision(5, "Acceptance against the model it would replace", spec.workload)
    decision.checks = _validity_checks(spec, cand) + _validity_checks(spec, base)[1:] + _bar_checks(spec, cand)
    decision.checks.append(
        Check(
            f"{candidate_name}: not worse than {spec.baseline_name} beyond the margin",
            delta["ci95"][0] >= -acc.noninferiority_margin,
            f"Accuracy difference {delta['delta'] * 100:+.1f} points (95% paired interval "
            f"{delta['ci95'][0] * 100:+.1f} to {delta['ci95'][1] * 100:+.1f}); "
            f"the lower end must be at or above {-acc.noninferiority_margin * 100:+.1f}.",
        )
    )
    for label, rule in spec.classes.items():
        if rule.min_recall is None:
            continue
        c_rec, b_rec = cand.classes[label].recall, base.classes[label].recall
        if c_rec is None or b_rec is None:
            continue
        decision.checks.append(
            Check(
                f"{candidate_name}: {label} recall not below {spec.baseline_name} beyond the margin",
                c_rec >= b_rec - acc.noninferiority_margin,
                f"{candidate_name} {_pct(c_rec)} against {spec.baseline_name} {_pct(b_rec)}; "
                f"margin {_pct(acc.noninferiority_margin)}.",
            )
        )
    decision.metrics = {"candidate": cand.to_dict(), "baseline": base.to_dict(), "paired": delta,
                        "slices": _slices(spec, items, cand, base)}
    decision.decide()
    decision.findings.extend(_diagnose(spec, items, cand, base, delta))
    return decision


def _slices(spec: GateSpec, items: list[dict], cand: RunMetrics, base: RunMetrics) -> list[dict]:
    out = []
    for field_name in spec.report_slices:
        ids = [i["id"] for i in items if i.get(field_name) is True]
        if not ids:
            continue
        out.append({
            "slice": field_name,
            "items": len(ids),
            "candidate_correct": sum(cand.hits[i] for i in ids),
            "baseline_correct": sum(base.hits[i] for i in ids),
        })
    return out


def _diagnose(spec: GateSpec, items: list[dict], cand: RunMetrics, base: RunMetrics, delta: dict) -> list[str]:
    """Point at model, data, or workflow. Plain statements from the counts only."""
    notes = []
    if cand.parsed < cand.total:
        notes.append(
            f"Workflow: {cand.total - cand.parsed} candidate replies did not parse to one label. "
            "Check the prompt template and output format before retraining."
        )
    if delta.get("both_wrong"):
        notes.append(
            f"Data: {delta['both_wrong']} items are wrong for both models. Review them for label errors "
            "or task-definition gaps; confirmed ones become test cases."
        )
    weakest = [
        (label, c.recall)
        for label, c in cand.classes.items()
        if c.support and c.recall is not None and base.classes[label].recall is not None
        and c.recall < base.classes[label].recall - spec.acceptance.noninferiority_margin
    ]
    if weakest:
        names = ", ".join(f"{label} ({_pct(rec)})" for label, rec in weakest)
        notes.append(
            f"Model or data coverage: the candidate trails the baseline on {names}. "
            "Add training examples for those labels, or keep them on the larger model."
        )
    if delta.get("baseline_only_right"):
        notes.append(
            f"{delta['baseline_only_right']} items only the baseline got right; "
            f"{delta.get('candidate_only_right', 0)} items only the candidate got right."
        )
    return notes
