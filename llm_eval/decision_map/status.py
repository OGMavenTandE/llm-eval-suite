"""Map a measurement to met, partial, gap, or not assessed.

Met requires the 95% Wilson lower bound to clear the bar, a large enough
sample, and a result inside the age limit. Point metrics such as latency
have no interval. The row status is the worst item: gap, then partial,
then not assessed, then met.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timezone
from typing import Literal

from llm_eval.decision_map.metrics import Observation
from llm_eval.decision_map.schema import (
    ComparisonOp,
    EvidenceRequirement,
    EvidenceStatus,
    ResolvedEvidence,
)

ReadingKind = Literal["proportion", "point", "manual"]

_Z_95 = 1.959963984540054
_RANK = {"gap": 3, "partial": 2, "not_assessed": 1, "met": 0}
_NOT_ASSESSED = {
    "not_run": "not run",
    "withheld": "withheld",
    "fixture": "fixture",
    "absent": "metric absent",
}


def wilson_interval(successes: int, sample_count: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion. Bounds stay inside 0 and 1."""
    if sample_count <= 0:
        raise ValueError("Wilson interval needs at least one sample")
    if abs(confidence - 0.95) > 1e-9:
        raise ValueError("confidence must be 0.95")
    if successes < 0 or successes > sample_count:
        raise ValueError("successes must be between 0 and the sample count")
    z = _Z_95
    p = successes / sample_count
    z2 = z * z
    denom = 1.0 + z2 / sample_count
    center = (p + z2 / (2.0 * sample_count)) / denom
    margin = z * math.sqrt(p * (1.0 - p) / sample_count + z2 / (4.0 * sample_count * sample_count)) / denom
    low = max(0.0, center - margin)
    high = min(1.0, center + margin)
    return low, high


def worst_status(statuses: list[EvidenceStatus]) -> EvidenceStatus:
    if not statuses:
        return "not_assessed"
    return max(statuses, key=lambda status: _RANK[status])


def as_utc(value: datetime | date | None) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)


def age_in_days(completed: datetime, now: datetime) -> int:
    return (now.date() - completed.astimezone(timezone.utc).date()).days


def not_measured(evidence: EvidenceRequirement, reason: str) -> ResolvedEvidence:
    """A scored item with no usable run. Kind follows the metric path when it is known."""
    from llm_eval.decision_map.metrics import declared_kind

    kind: ReadingKind = "proportion"
    if evidence.metric:
        kind = declared_kind(evidence.metric)
    return _base(evidence, kind=kind).model_copy(update={"status": "not_assessed", "reason": reason})


def judge_manual(
    evidence: EvidenceRequirement,
    *,
    now: datetime,
    run_bits: dict | None = None,
) -> ResolvedEvidence:
    attestation = evidence.attestation
    base = _base(evidence, kind="manual")
    if attestation is None:
        return base.model_copy(update={"status": "not_assessed", "reason": "no signed attestation"})
    update = {
        "attestation_who": attestation.who,
        "attestation_date": attestation.date.isoformat(),
        "attestation_expiry": None if attestation.expiry is None else attestation.expiry.isoformat(),
    }
    if attestation.date > now.date():
        update.update({"status": "partial", "reason": "attestation date is after the as-of date"})
        return base.model_copy(update=update)
    if attestation.expiry is not None and now.date() > attestation.expiry:
        if attestation.status == "gap":
            update.update({"status": "gap", "reason": f"attestation expired {attestation.expiry.isoformat()}"})
        else:
            update.update({"status": "partial", "reason": f"attestation expired {attestation.expiry.isoformat()}"})
        return base.model_copy(update=update)
    reason = attestation.note.strip()
    update.update({"status": attestation.status, "reason": reason})
    if run_bits:
        update.update(run_bits)
    return base.model_copy(update=update)


def judge_observation(
    evidence: EvidenceRequirement,
    observation: Observation,
    *,
    min_samples: int,
    max_age_days: int,
    confidence: float,
    now: datetime,
    completed_at: datetime | None,
    run_bits: dict | None = None,
) -> ResolvedEvidence:
    base = _base(evidence, kind=observation.kind)
    update: dict = {"metric": evidence.metric}
    if run_bits:
        update.update(run_bits)
    if observation.availability != "ok":
        detail = observation.detail or _NOT_ASSESSED[observation.availability]
        update.update({"status": "not_assessed", "reason": detail})
        return base.model_copy(update=update)
    if observation.value is None:
        update.update({"status": "not_assessed", "reason": observation.detail or "metric absent"})
        return base.model_copy(update=update)

    value = float(observation.value)
    sample_count = observation.sample_count
    update.update({"value": value, "sample_count": sample_count, "successes": observation.successes})
    if completed_at is not None:
        update["age_days"] = age_in_days(completed_at, now)
        update["completed_at"] = completed_at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    gap: list[str] = []
    partial: list[str] = []
    low: float | None = None
    high: float | None = None
    op = evidence.op
    threshold = evidence.threshold
    assert op is not None and threshold is not None

    if observation.kind == "proportion":
        if sample_count is None or sample_count <= 0 or observation.successes is None:
            update.update({"status": "not_assessed", "reason": "metric absent"})
            return base.model_copy(update=update)
        low, high = wilson_interval(observation.successes, sample_count, confidence)
        update.update({"ci_low": low, "ci_high": high})
        if _interval_fails(low, high, op, threshold):
            gap.append(f"CI {low:.2f}-{high:.2f}")
        elif _straddles(low, high, op, threshold, value):
            partial.append(f"CI {low:.2f}-{high:.2f}")

    _value_reasons(value, op, threshold, evidence.partial_floor, observation.kind, gap, partial)

    if sample_count is None:
        partial.append("sample count missing")
    elif sample_count < min_samples:
        partial.append(f"n={sample_count} < {min_samples}")

    if completed_at is None:
        partial.append("completion time missing")
    else:
        age = age_in_days(completed_at, now)
        if age < 0:
            partial.append("completion time is after the as-of date")
        elif age > max_age_days:
            unit = "day" if age == 1 else "days"
            partial.append(f"{age} {unit} old")

    if gap:
        status: EvidenceStatus = "gap"
        reason = _join(gap + partial)
    elif partial:
        status = "partial"
        reason = _join(partial)
    else:
        status = "met"
        reason = _met_reason(evidence, observation, value, sample_count, low)
    update.update({"status": status, "reason": reason})
    return base.model_copy(update=update)


def combine_reasons(items: list[ResolvedEvidence]) -> str:
    pool = [item for item in items if item.status != "met"] or list(items)
    parts: list[str] = []
    for item in pool:
        text = item.reason.strip()
        if text and text not in parts:
            parts.append(text)
    return "; ".join(parts)


def _base(evidence: EvidenceRequirement, *, kind: ReadingKind) -> ResolvedEvidence:
    return ResolvedEvidence(
        id=evidence.id,
        label=evidence.label,
        status="not_assessed",
        manual=evidence.manual,
        metric=evidence.metric,
        op=evidence.op,
        threshold=evidence.threshold,
        partial_floor=evidence.partial_floor,
        kind=kind,
    )


def _join(parts: list[str]) -> str:
    unique: list[str] = []
    for part in parts:
        if part and part not in unique:
            unique.append(part)
    return "; ".join(unique)


def _clears(value: float, op: ComparisonOp, threshold: float) -> bool:
    if op == ">=":
        return value >= threshold
    if op == ">":
        return value > threshold
    if op == "<=":
        return value <= threshold
    return value < threshold


def _interval_fails(low: float, high: float, op: ComparisonOp, threshold: float) -> bool:
    """True when every point in the interval misses the bar."""
    if op in {">=", ">"}:
        return not _clears(high, op, threshold)
    return not _clears(low, op, threshold)


def _straddles(low: float, high: float, op: ComparisonOp, threshold: float, value: float) -> bool:
    if not _clears(value, op, threshold):
        return False
    if op in {">=", ">"}:
        return not _clears(low, op, threshold)
    return not _clears(high, op, threshold)


def _fmt(value: float, kind: str) -> str:
    if kind == "proportion":
        return f"{value:.2f}"
    if abs(value - round(value)) < 1e-6:
        return str(int(round(value)))
    return f"{value:.2f}"


def _value_reasons(
    value: float,
    op: ComparisonOp,
    threshold: float,
    floor: float | None,
    kind: str,
    gap: list[str],
    partial: list[str],
) -> None:
    shown = _fmt(value, kind)
    bar = _fmt(threshold, kind)
    if floor is None:
        if not _clears(value, op, threshold):
            side = "below" if op in {">=", ">"} else "above"
            gap.append(f"{shown} is {side} {bar}")
        return
    floor_text = _fmt(floor, kind)
    if not _clears(value, op, floor):
        side = "below" if op in {">=", ">"} else "above"
        gap.append(f"{shown} is {side} {floor_text}")
    elif not _clears(value, op, threshold):
        partial.append(f"{shown} is between {floor_text} and {bar}")


def _met_reason(
    evidence: EvidenceRequirement,
    observation: Observation,
    value: float,
    sample_count: int | None,
    low: float | None,
) -> str:
    count = "" if sample_count is None else f", n={sample_count}"
    if observation.kind == "proportion" and low is not None and evidence.op in {">=", ">"}:
        return f"lower bound {low:.2f}{count}"
    if observation.kind == "proportion" and observation.sample_count and evidence.op in {"<=", "<"}:
        return f"upper bound clears{count}"
    tail = (evidence.metric or "").split(".")[-1]
    if tail in {"p50", "p95"}:
        return f"{tail} {_fmt(value, 'point')}{count}"
    return f"{_fmt(value, observation.kind)}{count}"
