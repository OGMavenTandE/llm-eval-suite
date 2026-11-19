"""Gate 6: ATO-aware continuous test and evaluation.

Run after cutover, on a schedule and whenever something changes. It compares
a reference run (the one that passed Gate 5) with a current run and checks
three things, each only when the inputs allow it:

- Accuracy drop on a labeled monitoring set (needs labels).
- Shift in the predicted-label mix, as a population stability index (PSI).
  This works on unlabeled production traffic.
- Agreement between two model versions on the same items, for a base-model
  or adapter version change.
"""

from __future__ import annotations

import math

from gates.acceptance import Check, GateDecision
from gates.metrics import paired_bootstrap_delta, score_run
from gates.spec import GateSpec

_PSI_FLOOR = 1e-4  # stands in for an empty bucket so the log stays finite


def label_mix(preds: dict[str, dict], labels: list[str]) -> dict[str, float]:
    counts = {label: 0 for label in labels}
    total = 0
    for row in preds.values():
        if row.get("prediction") in counts:
            counts[row["prediction"]] += 1
            total += 1
    return {label: (counts[label] / total if total else 0.0) for label in labels}


def psi(reference: dict[str, float], current: dict[str, float]) -> float:
    value = 0.0
    for label in reference:
        r = max(reference[label], _PSI_FLOOR)
        c = max(current.get(label, 0.0), _PSI_FLOOR)
        value += (c - r) * math.log(c / r)
    return value


def run_drift(
    spec: GateSpec,
    reference_preds: dict[str, dict],
    current_preds: dict[str, dict],
    *,
    items: list[dict] | None = None,
    reference_name: str = "reference",
    current_name: str = "current",
) -> GateDecision:
    policy = spec.drift
    decision = GateDecision(6, "Continuous test and evaluation (drift)", spec.workload)
    metrics: dict = {}

    ref_mix = label_mix(reference_preds, spec.labels)
    cur_mix = label_mix(current_preds, spec.labels)
    shift = psi(ref_mix, cur_mix)
    metrics["label_mix"] = {"reference": ref_mix, "current": cur_mix, "psi": shift}
    decision.checks.append(
        Check(
            "Predicted-label mix is stable",
            shift <= policy.max_psi,
            f"PSI {shift:.3f} between the {reference_name} and {current_name} label mix; limit {policy.max_psi:.3f}. "
            "A shift can be real change in traffic, not a fault, so read it with the accuracy check.",
        )
    )

    shared = sorted(set(reference_preds) & set(current_preds))
    if shared:
        agree = sum(1 for i in shared if reference_preds[i].get("prediction") == current_preds[i].get("prediction"))
        rate = agree / len(shared)
        metrics["agreement"] = {"items": len(shared), "agree": agree, "rate": rate}
        decision.checks.append(
            Check(
                "Versions agree on the same items",
                rate >= policy.min_agreement,
                f"{agree} of {len(shared)} shared items got the same label ({rate * 100:.1f}%); "
                f"limit {policy.min_agreement * 100:.1f}%.",
            )
        )

    if items:
        ref = score_run(reference_name, items, reference_preds, spec.labels)
        cur = score_run(current_name, items, current_preds, spec.labels)
        delta = paired_bootstrap_delta(cur.hits, ref.hits, resamples=spec.acceptance.bootstrap_resamples,
                                       seed=spec.acceptance.seed)
        metrics["reference"] = ref.to_dict()
        metrics["current"] = cur.to_dict()
        metrics["paired"] = delta
        decision.checks.append(
            Check(
                "Accuracy has not dropped beyond the limit",
                delta["ci95"][0] >= -policy.max_accuracy_drop,
                f"{current_name} {cur.accuracy * 100:.1f}% against {reference_name} {ref.accuracy * 100:.1f}% "
                f"(difference {delta['delta'] * 100:+.1f} points, 95% paired interval "
                f"{delta['ci95'][0] * 100:+.1f} to {delta['ci95'][1] * 100:+.1f}); "
                f"the lower end must be at or above {-policy.max_accuracy_drop * 100:+.1f}.",
            )
        )
    else:
        decision.findings.append("No labeled monitoring set was given, so accuracy was not checked this run.")

    decision.metrics = metrics
    decision.decide()
    if decision.verdict != "GO":
        decision.findings.append(
            "Hold the change or roll back, then find the cause: a model or version change, a data shift, "
            "or a workflow change. Save confirmed failures as test cases."
        )
    return decision
