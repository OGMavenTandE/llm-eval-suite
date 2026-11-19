"""Evaluation gates: spec, metrics, acceptance, audit, router, drift, reports, CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gates import GO, INVALID, NO_GO
from gates.acceptance import gate2_baseline, gate5_acceptance
from gates.audit import AuditConfig, TaskType, run_audit, sort_row
from gates.cli import main as gates_main
from gates.data import load_items, parse_label
from gates.demo import run_demo
from gates.drift import psi, run_drift
from gates.metrics import paired_bootstrap_delta, score_run, wilson_interval
from gates.report import render_html, render_markdown
from gates.router import run_router_harness
from gates.spec import GateSpec, load_spec

EXAMPLE = Path(__file__).resolve().parents[2] / "examples" / "message_triage"
LABELS = ["A", "B", "C"]


def make_spec(**acceptance) -> GateSpec:
    return GateSpec.model_validate(
        {
            "workload": "test",
            "labels": LABELS,
            "classes": {"A": {"min_recall": 0.9, "cost_of_miss": "high"}},
            "acceptance": {"min_items": 10, "min_parsed": 0.9, "min_accuracy": 0.8,
                           "noninferiority_margin": 0.05, "bootstrap_resamples": 500, **acceptance},
        }
    )


def make_items(n: int = 60) -> list[dict]:
    return [{"id": f"i{k}", "text": f"t{k}", "label": LABELS[k % 3]} for k in range(n)]


def preds_from(items, wrong: set[str] = frozenset(), unparsed: set[str] = frozenset()) -> dict[str, dict]:
    out = {}
    for item in items:
        if item["id"] in unparsed:
            pred = None
        elif item["id"] in wrong:
            pred = LABELS[(LABELS.index(item["label"]) + 1) % 3]
        else:
            pred = item["label"]
        out[item["id"]] = {"id": item["id"], "prediction": pred}
    return out


# --- spec and parsing ---------------------------------------------------------

def test_spec_rejects_unknown_class_rule():
    with pytest.raises(ValueError):
        GateSpec.model_validate({"workload": "x", "labels": ["A", "B"], "classes": {"Z": {"min_recall": 0.5}}})


def test_example_spec_loads():
    spec = load_spec(EXAMPLE / "spec.yaml")
    assert spec.labels == ["FLASH", "IMMEDIATE", "PRIORITY", "ROUTINE"]
    assert "{text}" in spec.prompt_template


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("FLASH", "FLASH"),
        ("  flash.\n", "FLASH"),
        ("Precedence: IMMEDIATE", "IMMEDIATE"),
        ("<think>could be FLASH</think>ROUTINE", "ROUTINE"),
        ("PRIORITY or IMMEDIATE", None),  # a hedge is not a hit
        ("", None),
        (None, None),
        ("unclear", None),
    ],
)
def test_parse_label(reply, expected):
    assert parse_label(reply, ["FLASH", "IMMEDIATE", "PRIORITY", "ROUTINE"]) == expected


# --- metrics ------------------------------------------------------------------

def test_wilson_interval_bounds():
    low, high = wilson_interval(9, 10)
    assert 0.5 < low < 0.9 < high <= 1.0
    assert wilson_interval(0, 0) == (0.0, 0.0)


def test_unparsed_counts_as_wrong():
    items = make_items(30)
    run = score_run("m", items, preds_from(items, unparsed={"i0", "i1"}), LABELS)
    assert run.total == 30 and run.parsed == 28 and run.correct == 28
    assert run.confusion["A"]["UNPARSED"] == 1


def test_paired_bootstrap_identical_models_is_zero():
    hits = {f"i{k}": k % 4 != 0 for k in range(80)}
    delta = paired_bootstrap_delta(hits, dict(hits), resamples=300, seed=1)
    assert delta["delta"] == 0 and delta["ci95"] == [0.0, 0.0]


# --- gates 2 and 5 ------------------------------------------------------------

def test_gate2_go_and_no_go():
    spec = make_spec()
    items = make_items()
    assert gate2_baseline(spec, items, preds_from(items)).verdict == GO
    wrong = {i["id"] for i in items[:20]}
    assert gate2_baseline(spec, items, preds_from(items, wrong=wrong)).verdict == NO_GO


def test_gate5_go_when_candidate_matches():
    spec = make_spec()
    items = make_items()
    base = preds_from(items, wrong={"i4", "i5"})
    decision = gate5_acceptance(spec, items, preds_from(items, wrong={"i4", "i5"}), base)
    assert decision.verdict == GO


def test_gate5_no_go_when_candidate_misses_critical_class():
    spec = make_spec()
    items = make_items()
    a_items = [i["id"] for i in items if i["label"] == "A"]
    decision = gate5_acceptance(spec, items, preds_from(items, wrong=set(a_items[:5])), preds_from(items))
    assert decision.verdict == NO_GO
    failed = [c.name for c in decision.checks if not c.passed]
    assert any("A recall" in name for name in failed)


def test_gate5_invalid_when_replies_do_not_parse():
    spec = make_spec()
    items = make_items()
    unparsed = {i["id"] for i in items[:15]}
    decision = gate5_acceptance(spec, items, preds_from(items, unparsed=unparsed), preds_from(items))
    assert decision.verdict == INVALID


def test_gate5_invalid_when_too_few_items():
    spec = make_spec(min_items=100)
    items = make_items(30)
    assert gate5_acceptance(spec, items, preds_from(items), preds_from(items)).verdict == INVALID


# --- gates 3 and 4 ------------------------------------------------------------

def audit_config() -> AuditConfig:
    return AuditConfig(
        min_rows=5,
        task_types={
            "triage": TaskType(keywords=["precedence"], route="small", cost_of_wrong="high"),
            "format": TaskType(keywords=["convert"], route="deterministic", cost_of_wrong="low"),
            "planning": TaskType(keywords=["course of action"], route="large", cost_of_wrong="high"),
        },
    )


def test_hard_triage_goes_large():
    row = {"prompt": "Assign a precedence, explain why, and recommend actions."}
    assert sort_row(row, audit_config()).route == "large"
    assert sort_row({"prompt": "Assign a precedence to: gym closed."}, audit_config()).route == "small"


def test_audit_uses_logged_tokens_and_maps_routes():
    rows = [{"prompt": "Assign a precedence to this", "tokens_in": 100, "tokens_out": 5}] * 6 + [
        {"prompt": "Convert 3pm to 24-hour", "tokens_in": 20, "tokens_out": 5}
    ] * 2
    decision = run_audit(rows, audit_config())
    assert decision.verdict == GO
    assert decision.metrics["route_calls"] == {"deterministic": 2, "small": 6, "large": 0}
    assert decision.metrics["total_tokens"] == 6 * 105 + 2 * 25
    assert decision.metrics["tokens_estimated"] is False


def test_audit_invalid_on_small_log():
    assert run_audit([{"prompt": "convert"}], audit_config()).verdict == INVALID


def test_router_flags_unsafe_small_route():
    config = audit_config()
    cases = [
        {"id": "ok", "prompt": "Assign a precedence to: gym closed.", "expected_route": "small"},
        {"id": "s1", "stress": True, "prompt": "Assign a precedence to this one", "expected_route": "large"},
    ]
    decision = run_router_harness(cases, config)
    assert decision.verdict == NO_GO
    assert decision.metrics["unsafe"] == 1


# --- gate 6 -------------------------------------------------------------------

def test_psi_zero_for_same_mix():
    mix = {"A": 0.5, "B": 0.3, "C": 0.2}
    assert psi(mix, dict(mix)) == pytest.approx(0.0)
    assert psi(mix, {"A": 0.1, "B": 0.1, "C": 0.8}) > 0.2


def test_drift_catches_accuracy_drop():
    spec = make_spec()
    items = make_items(90)
    ref = preds_from(items)
    cur = preds_from(items, wrong={i["id"] for i in items[:15]})
    decision = run_drift(spec, ref, cur, items=items)
    assert decision.verdict == NO_GO
    stable = run_drift(spec, ref, dict(ref), items=items)
    assert stable.verdict == GO


def test_drift_without_labels_notes_skipped_accuracy():
    spec = make_spec()
    items = make_items()
    decision = run_drift(spec, preds_from(items), preds_from(items))
    assert decision.verdict == GO
    assert any("accuracy was not checked" in f for f in decision.findings)


# --- reports, demo, CLI -------------------------------------------------------

def test_report_renders_verdict_and_notice():
    spec = make_spec()
    items = make_items()
    decision = gate5_acceptance(spec, items, preds_from(items), preds_from(items))
    decision.notice = "SIMULATED DEMO"
    html = render_html(decision, generated_at="2026-11-18 00:00 UTC")
    assert html.startswith("<!DOCTYPE html>") and 'class="report-doc"' in html
    assert "SIMULATED DEMO" in html and "GO: this gate passed" in html
    assert "/static/fonts/" not in html  # fonts are inlined for offline use
    md = render_markdown(decision)
    assert md.startswith("# Gate 5") and "> SIMULATED DEMO" in md


def test_demo_packet_marks_every_page_simulated(tmp_path):
    index = run_demo(tmp_path, generated_at="2026-11-18 00:00 UTC")
    text = index.read_text(encoding="utf-8")
    for gate in (2, 3, 4, 5, 6):
        assert f"Gate {gate}" in text
        data = json.loads((tmp_path / f"gate{gate}.json").read_text(encoding="utf-8"))
        assert data["notice"].startswith("SIMULATED DEMO")
        assert "SIMULATED DEMO" in (tmp_path / f"gate{gate}.html").read_text(encoding="utf-8")


def test_example_data_splits_do_not_overlap():
    train = load_items(EXAMPLE / "items.jsonl", split="train")
    test = load_items(EXAMPLE / "items.jsonl", split="test")
    assert train and test
    train_text = {i["text"] for i in train}
    assert {i["id"] for i in train}.isdisjoint({i["id"] for i in test})
    assert all(i.get("source") == "synthetic" for i in train + test)
    assert len(test) >= load_spec(EXAMPLE / "spec.yaml").acceptance.min_items
    assert not any(i["text"] in train_text for i in test)  # no message is in both splits
    assert not any(i["heldout_phrasing"] for i in train)
    assert any(i["heldout_phrasing"] for i in test)


def test_cli_accept_exit_codes(tmp_path):
    spec = tmp_path / "spec.yaml"
    spec.write_text(
        "workload: t\nlabels: [A, B, C]\nacceptance: {min_items: 10, min_parsed: 0.9, min_accuracy: 0.5, "
        "bootstrap_resamples: 200}\n",
        encoding="utf-8",
    )
    items = make_items(30)
    items_path = tmp_path / "items.jsonl"
    items_path.write_text("".join(json.dumps({**i, "split": "test"}) + "\n" for i in items), encoding="utf-8")
    good = tmp_path / "good.jsonl"
    good.write_text("".join(json.dumps({"id": i["id"], "reply": i["label"]}) + "\n" for i in items), encoding="utf-8")
    bad = tmp_path / "bad.jsonl"
    bad.write_text("".join(json.dumps({"id": i["id"], "reply": "no idea"}) + "\n" for i in items), encoding="utf-8")
    common = ["accept", "--spec", str(spec), "--items", str(items_path), "--baseline", str(good), "--out"]
    assert gates_main(common + [str(tmp_path / "go"), "--candidate", str(good)]) == 0
    assert gates_main(common + [str(tmp_path / "invalid"), "--candidate", str(bad)]) == 2
    assert (tmp_path / "go" / "gate5.html").is_file()
    record = json.loads((tmp_path / "go" / "gate5.json").read_text(encoding="utf-8"))
    assert set(record["inputs"]) == {"spec", "items", "candidate predictions", "baseline predictions"}


def test_train_lora_needs_peft():
    pytest.importorskip("peft")
    from gates import train_lora  # noqa: F401


def test_gate5_reports_slices():
    spec = make_spec()
    spec.report_slices = ["unseen"]
    items = [{**i, "unseen": k < 12} for k, i in enumerate(make_items())]
    decision = gate5_acceptance(spec, items, preds_from(items, wrong={"i0"}), preds_from(items))
    assert decision.metrics["slices"] == [
        {"slice": "unseen", "items": 12, "candidate_correct": 11, "baseline_correct": 12}
    ]
    assert "Slices" in render_html(decision)


def test_predict_records_replies_and_errors(tmp_path):
    from llm_eval.models.base import BaseModel, ModelResponse
    from gates.predict import predict

    class Stub(BaseModel):
        def generate(self, prompt: str, **kwargs) -> ModelResponse:
            if "t1" in prompt:
                raise TimeoutError("slow")
            return ModelResponse(text="B" if "t2" in prompt else "A", latency_ms=1.0, tokens_used=1)

    spec = make_spec()
    items = make_items(3)
    rows = predict(spec, items, Stub("stub", {}), out_path=tmp_path / "p.jsonl", progress=None)
    assert [r["prediction"] for r in rows] == ["A", None, "B"]
    assert rows[1]["error"].startswith("TimeoutError")
    assert (tmp_path / "p.jsonl").read_text(encoding="utf-8").count("\n") == 3


def test_predict_params_are_deterministic_per_provider():
    from gates.cli import predict_params

    assert predict_params("ollama", 16) == {"options": {"num_predict": 16, "temperature": 0}}
    assert predict_params("openai", 16) == {"max_new_tokens": 16, "temperature": 0}
    assert predict_params("hf", 16) == {"max_new_tokens": 16}
