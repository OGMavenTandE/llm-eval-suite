"""Resolve a map against click-through runs. Evidence status only."""

from __future__ import annotations

import json
from datetime import datetime

from llm_eval.decision_map.load import LoadedRun, MapError, merged_filters, newest, qualifies
from llm_eval.decision_map.metrics import MetricPathError, read_metric
from llm_eval.decision_map.providers import MetricRegistry, default_registry
from llm_eval.decision_map.schema import (
    Decision,
    DecisionMap,
    ResolvedDecision,
    ResolvedMap,
    RunProvenance,
    contains_forbidden,
)
from llm_eval.decision_map.status import (
    as_utc,
    combine_reasons,
    judge_manual,
    judge_observation,
    not_measured,
    worst_status,
)

CAVEAT = "Met means the 95% lower bound clears the bar on the tested conditions only."


def resolve_map(
    decision_map: DecisionMap,
    runs: list[LoadedRun],
    *,
    now: datetime | None = None,
    registry: MetricRegistry | None = None,
) -> ResolvedMap:
    registry = default_registry if registry is None else registry
    as_of = as_utc(now)
    used: dict[str, LoadedRun] = {}
    resolved_rows: list[ResolvedDecision] = []
    for decision in decision_map.decisions:
        evidence_rows = []
        for evidence in decision.evidence:
            if evidence.manual:
                evidence_rows.append(judge_manual(evidence, now=as_of))
                continue
            filters = merged_filters(decision_map.defaults.filters, evidence.filters)
            chosen = newest([run for run in runs if qualifies(run, filters)])
            if chosen is None:
                evidence_rows.append(not_measured(evidence, "no qualifying run"))
                continue
            used[chosen.run_id] = chosen
            try:
                observation = read_metric(chosen.record, chosen.items, evidence.metric or "", registry)
            except MetricPathError as exc:
                raise MapError([f"{decision.id}.{evidence.id}: {exc}"]) from exc
            evidence_rows.append(
                judge_observation(
                    evidence,
                    observation,
                    min_samples=evidence.min_samples if evidence.min_samples is not None else decision_map.defaults.min_samples,
                    max_age_days=_age_limit(decision_map, evidence),
                    confidence=decision_map.defaults.confidence,
                    now=as_of,
                    completed_at=chosen.completed_at,
                    run_bits=_run_bits(chosen),
                )
            )
        resolved_rows.append(_decision_row(decision, evidence_rows))
    resolved = ResolvedMap(
        system_name=decision_map.system.name,
        generated_at=as_of,
        as_of=as_of,
        confidence=decision_map.defaults.confidence,
        caveat=CAVEAT,
        decisions=resolved_rows,
        runs=[_provenance(run) for _, run in sorted(used.items())],
    )
    payload = json.dumps(resolved.model_dump(mode="json"))
    if contains_forbidden(payload):
        raise RuntimeError("resolved output contains wording this tool must not print")
    return resolved


def _age_limit(decision_map: DecisionMap, evidence) -> int:
    if evidence.max_result_age_days is not None:
        return evidence.max_result_age_days
    return decision_map.defaults.max_result_age_days


def _run_bits(run: LoadedRun) -> dict:
    connection = run.record.get("connection") or {}
    dataset = run.record.get("dataset") or {}
    return {
        "run_id": run.run_id,
        "model": connection.get("model") or None,
        "preset": run.record.get("preset") or None,
        "dataset_sha256": dataset.get("sha256") or None,
    }


def _decision_row(decision: Decision, evidence_rows) -> ResolvedDecision:
    status = worst_status([item.status for item in evidence_rows])
    return ResolvedDecision(
        id=decision.id,
        decision=decision.decision,
        status=status,
        reason=combine_reasons(evidence_rows),
        recommended_authority=decision.recommended_authority,
        evidence=evidence_rows,
    )


def _provenance(run: LoadedRun) -> RunProvenance:
    connection = run.record.get("connection") or {}
    dataset = run.record.get("dataset") or {}
    completed = run.completed_at
    return RunProvenance(
        run_id=run.run_id,
        model=connection.get("model") or None,
        precision=connection.get("precision") or run.record.get("precision") or None,
        preset=run.record.get("preset") or None,
        dataset_sha256=dataset.get("sha256") or None,
        completed_at=None if completed is None else completed.strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
