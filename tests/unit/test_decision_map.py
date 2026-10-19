"""Schema, metric paths, Wilson status, and the outside-metric hook."""

from __future__ import annotations

import ast
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from llm_eval.decision_map.load import MapError, load_map, load_runs
from llm_eval.decision_map.metrics import CATEGORY_KEYS, PUBLIC_SUITES, successes_from_rate
from llm_eval.decision_map.providers import (
    ExternalMetricProvider,
    MetricReading,
    MetricRegistry,
    default_registry,
    register_metric,
)
from llm_eval.decision_map.render import render_html
from llm_eval.decision_map.resolve import resolve_map
from llm_eval.decision_map.schema import (
    Attestation,
    Decision,
    DecisionMap,
    EvidenceFilters,
    EvidenceRequirement,
    MapDefaults,
    SystemInfo,
    trigger_text,
)
from llm_eval.decision_map.schema import AgeDaysTrigger, ManualTrigger, MetricDriftTrigger, ModelChangedTrigger
from llm_eval.decision_map.status import wilson_interval

AS_OF = datetime(2026, 10, 19, tzinfo=timezone.utc)
ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "llm_eval" / "decision_map"


def test_wilson_190_of_200():
    low, high = wilson_interval(190, 200)
    assert low == pytest.approx(0.9104, abs=1e-4)
    assert high == pytest.approx(0.9726, abs=1e-4)
    assert low >= 0.90


def test_category_names_match_the_scorecard():
    from llm_eval_suite.scoring import CATEGORIES, PUBLIC_BENCHMARK_SUITES

    assert CATEGORY_KEYS == {key for key, _label in CATEGORIES}
    assert PUBLIC_SUITES == set(PUBLIC_BENCHMARK_SUITES)


def test_schema_rejects_unknown_fields_and_bad_shapes():
    with pytest.raises(ValidationError):
        DecisionMap.model_validate(
            {
                "schema_version": 1,
                "system": {"name": "Demo"},
                "decisions": [
                    {
                        "id": "R1",
                        "decision": "Do it",
                        "risk": "low",
                        "authority_if_met": "Advisory only",
                        "evidence": [{"id": "e", "label": "Lab", "manual": True}],
                    }
                ],
            }
        )
    with pytest.raises(ValidationError):
        EvidenceRequirement(id="e", label="Lab", metric="scorecard.category.retrieval.pass_rate", op=">=", threshold=0.9, manual=True)
    with pytest.raises(ValidationError):
        EvidenceRequirement(id="e", label="Lab", metric="edge.latency_ms.p95", op="<=", threshold=100, partial_floor=50)
    with pytest.raises(ValidationError):
        DecisionMap.model_validate({"schema_version": 1, "system": {"name": "Demo"}, "defaults": {"confidence": 0.8}, "decisions": []})


def test_unknown_metric_names_the_row(tmp_path: Path):
    path = tmp_path / "map.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "system": {"name": "Demo"},
                "decisions": [
                    {
                        "id": "R3",
                        "decision": "Cue a crew",
                        "risk": "moderate",
                        "evidence": [
                            {
                                "id": "recall",
                                "label": "Held-out set",
                                "metric": "scorecard.category.not_a_category.pass_rate",
                                "op": ">=",
                                "threshold": 0.9,
                            }
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(MapError) as caught:
        load_map(path)
    assert "R3.recall" in str(caught.value)
    assert "unknown metric path" in str(caught.value)


def test_forbidden_wording_is_rejected(tmp_path: Path):
    path = tmp_path / "map.yaml"
    path.write_text(
        "schema_version: 1\nsystem: {name: Demo}\ndecisions:\n"
        "  - {id: R1, decision: Do it, risk: low, recommended_authority: 'Authority granted',\n"
        "     evidence: [{id: e, label: Lab, manual: true}]}\n",
        encoding="utf-8",
    )
    with pytest.raises(MapError, match="must not print"):
        load_map(path)


def test_duplicate_ids_and_drift_target(tmp_path: Path):
    path = tmp_path / "map.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "system": {"name": "Demo"},
                "decisions": [
                    {
                        "id": "R1",
                        "decision": "First",
                        "risk": "low",
                        "evidence": [
                            {"id": "a", "label": "One", "manual": True},
                            {"id": "a", "label": "Two", "manual": True},
                        ],
                        "triggers": [{"kind": "metric_drift", "evidence": "missing", "delta": 0.05}],
                    },
                    {
                        "id": "R1",
                        "decision": "Second",
                        "risk": "low",
                        "evidence": [{"id": "b", "label": "Three", "manual": True}],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(MapError) as caught:
        load_map(path)
    text = str(caught.value)
    assert "duplicate evidence id" in text
    assert "duplicate decision id" in text
    assert "missing" in text


def test_trigger_text_is_display_only():
    assert trigger_text(ManualTrigger(kind="manual", text="New camera")) == "New camera"
    assert trigger_text(AgeDaysTrigger(kind="age_days", value=30)) == "Every 30 days"
    assert trigger_text(ModelChangedTrigger(kind="model_changed")) == "Model change"
    assert trigger_text(MetricDriftTrigger(kind="metric_drift", evidence="recall", delta=0.05)) == "recall moves by 0.05"


def test_no_outside_metrics_are_shipped():
    assert default_registry.names() == ()
    for path in PACKAGE.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "playwright" not in source.casefold()
        assert "chromium" not in source.casefold()
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                lowered = node.name.casefold()
                assert "wer" not in lowered
                assert "perimeter" not in lowered


def test_plugin_hook_resolves_only_when_registered(tmp_path: Path):
    bare = _map(metric="plugin.perimeter.accuracy", threshold=0.9)
    with pytest.raises(MapError, match="no metric provider"):
        load_map(_dump(tmp_path, bare))

    class PerimeterAccuracy(ExternalMetricProvider):
        name = "perimeter"

        def paths(self) -> frozenset[str]:
            return frozenset({"accuracy"})

        def read(self, run: dict, items: list[dict], path: str) -> MetricReading | None:
            raw = (run.get("outside") or {}).get(path)
            if raw is None:
                return None
            return MetricReading(
                value=raw["value"],
                kind="proportion",
                sample_count=raw["n"],
                successes=raw["k"],
                source="live",
            )

    registry = MetricRegistry()
    register_metric(PerimeterAccuracy(), registry)
    decision_map = load_map(_dump(tmp_path / "ok", bare), registry)
    directory = _write_run(
        tmp_path / "runs",
        outside={"accuracy": {"value": 0.95, "n": 200, "k": 190}},
    )
    resolved = resolve_map(decision_map, load_runs(directory.parent), now=AS_OF, registry=registry)
    assert resolved.decisions[0].status == "met"
    assert resolved.decisions[0].evidence[0].ci_low == pytest.approx(0.9104, abs=1e-3)
    assert default_registry.names() == ()


def test_met_requires_lower_bound_samples_and_age(tmp_path: Path):
    runs = tmp_path / "runs"
    _write_run(
        runs,
        categories=[_category("hallucination_factuality", 0.95, 200)],
    )
    resolved = _resolve(tmp_path, runs, metric="scorecard.category.hallucination_factuality.pass_rate", threshold=0.90, min_samples=200)
    item = resolved.decisions[0].evidence[0]
    assert resolved.decisions[0].status == "met"
    assert item.successes == 190
    assert item.ci_low == pytest.approx(0.9104, abs=1e-3)
    assert "lower bound 0.91" in item.reason


def test_partial_when_interval_straddles_or_sample_is_short_or_stale(tmp_path: Path):
    runs = tmp_path / "straddle"
    _write_run(runs, categories=[_category("robustness", 0.92, 100)])
    straddling = _resolve(tmp_path, runs, metric="scorecard.category.robustness.pass_rate", threshold=0.90)
    assert straddling.decisions[0].status == "partial"
    assert straddling.decisions[0].evidence[0].reason == "CI 0.85-0.96"

    short = tmp_path / "short"
    _write_run(short, categories=[_category("robustness", 1.0, 60)])
    short_result = _resolve(tmp_path, short, metric="scorecard.category.robustness.pass_rate", threshold=0.90, min_samples=200)
    assert short_result.decisions[0].status == "partial"
    assert "n=60 < 200" in short_result.decisions[0].reason
    assert short_result.decisions[0].evidence[0].ci_low > 0.90

    stale = tmp_path / "stale"
    _write_run(
        stale,
        completed_at="2026-09-08T16:00:00+00:00",
        categories=[_category("robustness", 1.0, 200)],
    )
    stale_result = _resolve(tmp_path, stale, metric="scorecard.category.robustness.pass_rate", threshold=0.90)
    assert stale_result.decisions[0].status == "partial"
    assert stale_result.decisions[0].reason == "41 days old"


def test_gap_when_value_or_interval_misses_the_bar(tmp_path: Path):
    runs = tmp_path / "runs"
    _write_run(runs, categories=[_category("security_jailbreak", 0.80, 200)])
    resolved = _resolve(
        tmp_path,
        runs,
        metric="scorecard.category.security_jailbreak.pass_rate",
        threshold=0.98,
        partial_floor=0.90,
    )
    assert resolved.decisions[0].status == "gap"
    assert "0.80 is below 0.90" in resolved.decisions[0].reason
    assert "CI 0.74-0.85" in resolved.decisions[0].reason


def test_not_assessed_for_missing_fixture_and_invalid_runs(tmp_path: Path):
    missing = tmp_path / "missing"
    _write_run(missing, categories=[_category("retrieval", None, 0, source=None, status="not_run")])
    not_run = _resolve(tmp_path, missing, metric="scorecard.category.retrieval.pass_rate", threshold=0.9)
    assert not_run.decisions[0].status == "not_assessed"
    assert not_run.decisions[0].reason == "not run"

    fixture = tmp_path / "fixture"
    _write_run(fixture, categories=[_category("toxicity", 1.0, 10, source="fixture", status="fixture")])
    fixture_result = _resolve(tmp_path, fixture, metric="scorecard.category.toxicity.pass_rate", threshold=0.9)
    assert fixture_result.decisions[0].status == "not_assessed"
    assert fixture_result.decisions[0].reason == "fixture"

    invalid = tmp_path / "invalid"
    _write_run(invalid, validity="invalid", categories=[_category("robustness", 1.0, 200)])
    invalid_result = _resolve(tmp_path, invalid, metric="scorecard.category.robustness.pass_rate", threshold=0.9)
    assert invalid_result.decisions[0].status == "not_assessed"
    assert invalid_result.decisions[0].reason == "no qualifying run"


def test_live_rows_override_a_mixed_scorecard(tmp_path: Path):
    runs = tmp_path / "runs"
    directory = _write_run(runs, categories=[_category("hallucination_factuality", 1.0, 2, source="mixed")])
    rows = [
        {"category": "hallucination_factuality", "suite": "gsm8k", "source": "live", "passed": True, "counts_toward_score": True},
        {"category": "hallucination_factuality", "suite": "factcheck", "source": "fixture", "passed": True, "counts_toward_score": True},
        {"category": "hallucination_factuality", "suite": "factcheck", "source": "live", "passed": False, "counts_toward_score": True},
        {"category": "hallucination_factuality", "suite": "factcheck", "source": "live", "passed": True, "counts_toward_score": False},
    ]
    (directory / "items.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    resolved = _resolve(
        tmp_path,
        runs,
        metric="scorecard.category.hallucination_factuality.pass_rate",
        threshold=0.9,
        min_samples=1,
    )
    item = resolved.decisions[0].evidence[0]
    assert item.sample_count == 1
    assert item.successes == 0
    assert resolved.decisions[0].status == "gap"


def test_filters_and_newest_run(tmp_path: Path):
    runs = tmp_path / "runs"
    _write_run(
        runs,
        run_id="older",
        completed_at="2026-10-01T00:00:00+00:00",
        categories=[_category("robustness", 0.99, 200)],
    )
    _write_run(
        runs,
        run_id="newer",
        completed_at="2026-10-12T00:00:00+00:00",
        categories=[_category("robustness", 0.50, 200)],
    )
    _write_run(
        runs,
        run_id="other-preset",
        preset="quick",
        categories=[_category("robustness", 1.0, 200)],
    )
    resolved = _resolve(
        tmp_path,
        runs,
        metric="scorecard.category.robustness.pass_rate",
        threshold=0.90,
        preset="government_te",
    )
    item = resolved.decisions[0].evidence[0]
    assert item.run_id == "newer"
    assert item.value == pytest.approx(0.5)
    assert resolved.decisions[0].status == "gap"


def test_latency_has_no_interval(tmp_path: Path):
    runs = tmp_path / "runs"
    _write_run(runs, edge=_latency(80, n=50))
    met = _resolve(tmp_path, runs, metric="edge.latency_ms.p95", op="<=", threshold=100, min_samples=10)
    assert met.decisions[0].status == "met"
    assert met.decisions[0].evidence[0].ci_low is None
    assert met.decisions[0].evidence[0].reason == "p95 80, n=50"

    between = tmp_path / "between"
    _write_run(between, edge=_latency(150, n=50))
    partial = _resolve(
        tmp_path,
        between,
        metric="edge.latency_ms.p95",
        op="<=",
        threshold=100,
        partial_floor=200,
        min_samples=10,
    )
    assert partial.decisions[0].status == "partial"
    assert "between" in partial.decisions[0].reason

    above = tmp_path / "above"
    _write_run(above, edge=_latency(250, n=50))
    gap = _resolve(
        tmp_path,
        above,
        metric="edge.latency_ms.p95",
        op="<=",
        threshold=100,
        partial_floor=200,
        min_samples=10,
    )
    assert gap.decisions[0].status == "gap"
    assert "250 is above 200" in gap.decisions[0].reason


def test_manual_attestation_expires_to_partial():
    fresh = _manual("met", expiry=date(2026, 12, 1), note="")
    assert fresh.decisions[0].status == "met"
    expired = _manual("met", expiry=date(2026, 10, 1), note="Signed last month.")
    assert expired.decisions[0].status == "partial"
    assert "attestation expired 2026-10-01" in expired.decisions[0].reason
    still_gap = _manual("gap", expiry=date(2026, 10, 1), note="Failed the night trial.")
    assert still_gap.decisions[0].status == "gap"
    unsigned = resolve_map(_decision_map(_manual_evidence(None)), [], now=AS_OF)
    assert unsigned.decisions[0].status == "not_assessed"
    assert unsigned.decisions[0].reason == "no signed attestation"


def test_row_status_is_the_worst_item():
    decision_map = DecisionMap(
        schema_version=1,
        system=SystemInfo(name="Demo"),
        decisions=[
            Decision(
                id="D1",
                decision="Do the thing",
                risk="high",
                recommended_authority="Keep this phrase",
                evidence=[
                    EvidenceRequirement(
                        id="met",
                        label="Clear bar",
                        metric="scorecard.category.robustness.pass_rate",
                        op=">=",
                        threshold=0.5,
                        min_samples=10,
                    ),
                    EvidenceRequirement(id="open", label="No signature", manual=True),
                ],
            )
        ],
    )
    runs = _memory_runs([_category("robustness", 1.0, 20)])
    resolved = resolve_map(decision_map, runs, now=AS_OF)
    assert [item.status for item in resolved.decisions[0].evidence] == ["met", "not_assessed"]
    assert resolved.decisions[0].status == "not_assessed"
    assert resolved.decisions[0].recommended_authority == "Keep this phrase"


def test_authority_text_is_copied_and_never_swapped(tmp_path: Path):
    runs = tmp_path / "runs"
    _write_run(runs, categories=[_category("security_jailbreak", 0.2, 200)])
    decision_map = _map(
        metric="scorecard.category.security_jailbreak.pass_rate",
        threshold=0.9,
        recommended_authority="Keep this phrase",
    )
    resolved = resolve_map(decision_map, load_runs(runs), now=AS_OF)
    assert resolved.decisions[0].status == "gap"
    assert resolved.decisions[0].recommended_authority == "Keep this phrase"
    html = render_html(decision_map, resolved)
    assert "Keep this phrase" in html
    assert "None written" not in html
    assert "authority granted" not in html.casefold()
    assert "authority granted" not in resolved.model_dump_json().casefold()

    blank = decision_map.model_copy(deep=True)
    blank.decisions[0].recommended_authority = None
    blank_html = render_html(blank, resolve_map(blank, load_runs(runs), now=AS_OF))
    assert "<b>None written</b>" in blank_html
    assert "Keep this phrase" not in blank_html


def test_public_benchmark_and_dataset_count(tmp_path: Path):
    runs = tmp_path / "runs"
    _write_run(runs, row_count=240, suites=[{"name": "gsm8k", "source": "live", "status": "completed", "metrics": {"accuracy": 0.9, "n_scored": 50}}])
    public = _resolve(tmp_path, runs, metric="public.gsm8k.accuracy", threshold=0.75, min_samples=50)
    assert public.decisions[0].status == "met"
    counted = _resolve(tmp_path, runs, metric="dataset.sample_count", op=">=", threshold=100, min_samples=100)
    assert counted.decisions[0].status == "met"
    assert counted.decisions[0].evidence[0].ci_low is None
    assert successes_from_rate(0.95, 200) == 190


def _map(metric: str, threshold: float, **kwargs) -> DecisionMap:
    evidence = {
        "id": "e",
        "label": "Measured",
        "metric": metric,
        "op": kwargs.pop("op", ">="),
        "threshold": threshold,
    }
    for key in ("partial_floor", "min_samples"):
        if key in kwargs:
            evidence[key] = kwargs.pop(key)
    authority = kwargs.pop("recommended_authority", "Advisory only")
    preset = kwargs.pop("preset", None)
    defaults = MapDefaults(filters=EvidenceFilters(preset=preset)) if preset else MapDefaults()
    return DecisionMap(
        schema_version=1,
        system=SystemInfo(name="Demo"),
        defaults=defaults,
        decisions=[
            Decision(
                id="D1",
                decision="Do the thing",
                risk="moderate",
                recommended_authority=authority,
                evidence=[EvidenceRequirement(**evidence)],
            )
        ],
    )


def _dump(directory: Path, decision_map: DecisionMap) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "map.yaml"
    path.write_text(yaml.safe_dump(decision_map.model_dump(mode="json")), encoding="utf-8")
    return path


def _resolve(tmp_path: Path, runs: Path, **kwargs):
    decision_map = _map(**kwargs)
    return resolve_map(decision_map, load_runs(runs), now=AS_OF)


def _category(key: str, rate, sample_count: int, source="live", status="pass") -> dict:
    return {
        "category": key,
        "status": status,
        "pass_rate": rate,
        "sample_count": sample_count,
        "source": source,
    }


def _latency(value: float, n: int) -> dict:
    return {"requests": n, "latency_ms": {"p95": {"value": value, "reason": ""}, "n": n}}


def _write_run(runs: Path, run_id: str = "run-a", **kwargs) -> Path:
    categories = kwargs.pop("categories", [])
    edge = kwargs.pop("edge", {})
    suites = kwargs.pop("suites", [])
    row_count = kwargs.pop("row_count", 240)
    record = {
        "run_id": run_id,
        "status": kwargs.pop("status", "completed"),
        "validity": kwargs.pop("validity", "ok"),
        "preset": kwargs.pop("preset", "government_te"),
        "completed_at": kwargs.pop("completed_at", "2026-10-12T16:00:00+00:00"),
        "connection": {"model": "demo", "precision": "example"},
        "dataset": {"sha256": "ab", "row_count": row_count},
        "scorecard": {"categories": categories, "edge": edge},
        "suites": suites,
    }
    record.update(kwargs)
    directory = runs / run_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "run.json").write_text(json.dumps(record), encoding="utf-8")
    return directory


def _memory_runs(categories: list[dict]):
    directory_holder = Path("/tmp")
    record = {
        "run_id": "memory",
        "status": "completed",
        "validity": "ok",
        "preset": "government_te",
        "completed_at": "2026-10-12T16:00:00+00:00",
        "connection": {"model": "demo"},
        "dataset": {"sha256": "ab", "row_count": 20},
        "scorecard": {"categories": categories},
    }
    from llm_eval.decision_map.load import LoadedRun

    return [LoadedRun(directory=directory_holder, record=record, items=None)]


def _manual(status: str, expiry: date, note: str):
    return resolve_map(_decision_map(_manual_evidence(Attestation(who="A. Rivera", date=date(2026, 9, 1), signature="EXAMPLE", note=note, expiry=expiry, status=status))), [], now=AS_OF)


def _manual_evidence(attestation: Attestation | None) -> EvidenceRequirement:
    return EvidenceRequirement(id="e", label="Shadow run", manual=True, attestation=attestation)


def _decision_map(evidence: EvidenceRequirement) -> DecisionMap:
    return DecisionMap(
        schema_version=1,
        system=SystemInfo(name="Demo"),
        decisions=[Decision(id="D1", decision="Do the thing", risk="low", evidence=[evidence])],
    )
