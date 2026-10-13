"""Quantization delta report. No model server and no GPU."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_eval.edge.cli import main, parse_variant, require_standard_data, run_quant_compare
from llm_eval.edge.vram import prepare_variant_vram, unload_ollama_models
from llm_eval.edge.quant import build_delta_report, render_html, render_markdown
from llm_eval.edge.session import edge_concurrency
from llm_eval.models.base import BaseModel, ModelResponse
from llm_eval_suite.compare import compare_runs
from llm_eval_suite.presets import apply_per_suite, expand_preset, list_presets
from standard_bench.runner import resolve_limit


class ReplyModel(BaseModel):
    def __init__(self, name: str, reply: str):
        super().__init__(name, {})
        self.reply = reply

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        return ModelResponse(
            text=self.reply,
            latency_ms=20.0,
            tokens_used=4,
            metadata={"ttft_ms": 5.0, "output_tokens": 4, "eval_seconds": 0.1},
        )


def _edge(tokens_per_second, watts=None):
    def cell(value, reason=""):
        if value is None:
            return {"value": "n/a", "reason": reason or "not measured"}
        return {"value": value, "reason": ""}

    return {
        "ttft_ms": {"p50": cell(10.0), "p95": cell(12.0), "n": 2},
        "output_tokens_per_second": cell(tokens_per_second),
        "latency_ms": {"p50": cell(20.0), "p95": cell(30.0), "n": 2},
        "power_watts_average": cell(watts, "no NVIDIA GPU detected"),
        "power_watts_peak": cell(None if watts is None else watts + 10, "no NVIDIA GPU detected"),
        "joules_per_output_token": cell(None, "no NVIDIA GPU detected"),
        "peak_vram_mib": cell(None, "no NVIDIA GPU detected"),
        "peak_ram_mib": cell(8000.0),
    }


def _run(label, rate, *, invalid=False, tokens=80.0, watts=100.0, items=None):
    status = "invalid" if invalid else "completed"
    return {
        "label": label,
        "model": f"llama-{label}",
        "run": {
            "run_id": label,
            "status": status,
            "validity": "invalid" if invalid else "ok",
            "scorecard": {
                "categories": [
                    {
                        "category": "hallucination_factuality",
                        "label": "Hallucination / factuality",
                        "status": "withheld" if invalid else "pass",
                        "pass_rate": None if invalid else rate,
                        "sample_count": 5,
                        "source": "live",
                    }
                ]
            },
            "edge_metrics": _edge(tokens, watts),
        },
        "items": [] if items is None else items,
    }


def test_parse_variant_keeps_ollama_tags():
    assert parse_variant("q4=llama3.2:3b-instruct-q4_K_M") == ("q4", "llama3.2:3b-instruct-q4_K_M")
    with pytest.raises(ValueError):
        parse_variant("q4")


def test_accuracy_tolerance_and_power_deltas():
    report = build_delta_report(
        [_run("fp16", 1.0, tokens=80.0, watts=100.0), _run("q4", 0.95, tokens=120.0, watts=70.0)],
        baseline="fp16",
        tolerance=0.05,
    )
    assert report["verdict"] == "pass"
    variant = report["variants"][0]
    category = variant["accuracy"]["categories"][0]
    assert category["delta"] == -0.05
    assert category["status"] == "pass"
    benched = build_delta_report(
        [
            _run(
                "fp16",
                1.0,
                items=[{"suite": "gsm8k", "passed": True, "counts_toward_score": True}] * 4,
            ),
            _run(
                "q4",
                1.0,
                items=[{"suite": "gsm8k", "passed": index < 3, "counts_toward_score": True} for index in range(4)],
            ),
        ],
        baseline="fp16",
        tolerance=0.05,
    )
    benchmark = benched["variants"][0]["accuracy"]["benchmarks"][0]
    assert benchmark["name"] == "gsm8k"
    assert benchmark["delta"] == -0.25
    assert benchmark["status"] == "fail"
    assert benchmark["baseline_n"] == 4
    assert benchmark["variant_passed"] == 3
    assert benchmark["within_noise"] is True
    assert benchmark["note"] == "within +/- 1 item"
    assert benched["verdict"] == "fail"
    text = render_markdown(benched)
    html = render_html(benched)
    assert "3/4" in text
    assert "within +/- 1 item" in text
    assert "within +/- 1 item" in html
    assert "\u00b1" not in text
    assert "\u00b1" not in html
    speed = {row["name"]: row for row in variant["speed"]}
    assert speed["output_tokens_per_second"]["delta"] == 40.0
    power = {row["name"]: row for row in variant["power"]}
    assert power["power_watts_average"]["delta"] == -30.0
    assert power["joules_per_output_token"]["delta"] == "n/a"
    assert "NVIDIA" in power["joules_per_output_token"]["reason"]

    tight = build_delta_report(
        [_run("fp16", 1.0), _run("q4", 0.94)],
        baseline="fp16",
        tolerance=0.05,
    )
    assert tight["verdict"] == "fail"
    assert tight["variants"][0]["accuracy"]["categories"][0]["status"] == "fail"


def _bench_items(suite: str, passed: int, n: int) -> list[dict]:
    return [{"suite": suite, "passed": index < passed, "counts_toward_score": True} for index in range(n)]


def test_benchmark_noise_uses_item_counts_and_the_binomial_interval():
    one_item = build_delta_report(
        [
            _run("fp16", 1.0, items=_bench_items("gsm8k", 20, 20)),
            _run("q4", 1.0, items=_bench_items("gsm8k", 19, 20)),
        ],
        baseline="fp16",
        tolerance=0.02,
    )
    row = one_item["variants"][0]["accuracy"]["benchmarks"][0]
    assert row["delta"] == -0.05
    assert row["status"] == "fail"
    assert row["baseline_n"] == 20
    assert row["variant_n"] == 20
    assert row["baseline_passed"] == 20
    assert row["variant_passed"] == 19
    assert row["within_noise"] is True
    assert row["note"] == "within +/- 1 item"
    assert one_item["verdict"] == "fail"

    wide = build_delta_report(
        [
            _run("fp16", 1.0, items=_bench_items("mmlu_pro", 20, 20)),
            _run("q4", 1.0, items=_bench_items("mmlu_pro", 10, 20)),
        ],
        baseline="fp16",
        tolerance=0.05,
    )
    separated = wide["variants"][0]["accuracy"]["benchmarks"][0]
    assert separated["within_noise"] is False
    assert separated["note"] == ""
    assert separated["status"] == "fail"

    overlap = build_delta_report(
        [
            _run("fp16", 1.0, items=_bench_items("gsm8k", 27, 30)),
            _run("q4", 1.0, items=_bench_items("gsm8k", 24, 30)),
        ],
        baseline="fp16",
        tolerance=0.05,
    )
    noisy = overlap["variants"][0]["accuracy"]["benchmarks"][0]
    assert noisy["note"] == "within the 95% binomial interval"
    assert noisy["within_noise"] is True
    assert noisy["status"] == "fail"
    assert overlap["verdict"] == "fail"


def test_per_suite_caps_standard_quick_and_leaves_quant_quick_at_five():
    quick = expand_preset("standard_capability_quick")
    names = [row["name"] for row in quick["suites"]]
    assert names == ["mmlu_pro", "gsm8k", "math500", "ifeval", "humaneval_plus", "mbpp_plus"]
    capped = apply_per_suite(quick, 20)
    gsm = next(row for row in capped["suites"] if row["name"] == "gsm8k")
    assert gsm["limit"] == 20
    assert resolve_limit(gsm, "gsm8k") == 20
    assert capped["per_suite"] == 20
    smoke = apply_per_suite(expand_preset("quant_quick"), 20)
    assert smoke["suites"][0]["max_items"] == 5
    shrunk = apply_per_suite(expand_preset("quant_quick"), 2)
    assert shrunk["suites"][0]["max_items"] == 2
    with pytest.raises(ValueError):
        apply_per_suite(quick, 0)


def test_invalid_run_does_not_emit_an_accuracy_delta():
    report = build_delta_report(
        [_run("fp16", 1.0, invalid=True), _run("q4", 0.5)],
        baseline="fp16",
        tolerance=0.05,
    )
    assert report["verdict"] == "n/a"
    row = report["variants"][0]["accuracy"]["categories"][0]
    assert row["delta"] == "n/a"
    assert row["reason"] == "score withheld"
    text = render_markdown(report)
    html = render_html(report)
    assert "score withheld" in text
    assert "score withheld" in html
    assert "n/a" in text


def test_quant_quick_preset_and_connection_concurrency():
    expanded = expand_preset("quant_quick")
    assert [row["name"] for row in expanded["suites"]] == ["factcheck"]
    assert expanded["suites"][0]["max_items"] == 5
    assert expanded["edge"]["concurrency"] == 1
    assert any(row["id"] == "quant_quick" for row in list_presets())
    assert edge_concurrency(expanded, {"edge_concurrency": 4}) == 4
    assert edge_concurrency(expanded, {}) == 1


def test_cli_dry_run_writes_a_plan(tmp_path: Path):
    out = tmp_path / "quant"
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "quant-compare",
                "--dry-run",
                "--baseline",
                "fp16=llama3.2:3b-instruct-fp16",
                "--variant",
                "q4=llama3.2:3b-instruct-q4_K_M",
                "--out",
                str(out),
            ]
        )
    assert exc.value.code == 0
    plan = json.loads((out / "plan.json").read_text(encoding="utf-8"))
    assert plan["dry_run"] is True
    assert plan["baseline"] == "fp16"
    assert plan["variants"][1]["label"] == "q4"
    assert plan["note"] == "Dry run. No model was called."
    assert plan["per_suite"] is None
    assert not (out / "report.json").exists()

    standard = tmp_path / "standard"
    with pytest.raises(SystemExit) as standard_exc:
        main(
            [
                "quant-compare",
                "--dry-run",
                "--baseline",
                "fp16=llama3.2:3b-instruct-fp16",
                "--variant",
                "q4=llama3.2:3b-instruct-q4_K_M",
                "--preset",
                "standard_capability_quick",
                "--per-suite",
                "20",
                "--provider",
                "vllm",
                "--out",
                str(standard),
            ]
        )
    assert standard_exc.value.code == 0
    standard_plan = json.loads((standard / "plan.json").read_text(encoding="utf-8"))
    assert standard_plan["preset"] == "standard_capability_quick"
    assert standard_plan["per_suite"] == 20
    assert standard_plan["provider"] == "vllm"


def test_cli_run_writes_reports_for_a_stub_pair(tmp_path: Path):
    dataset = tmp_path / "facts.jsonl"
    dataset.write_text(
        json.dumps({"prompt": "Capital of France?", "expected_answer": "Paris"}) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "quant"

    def factory(profile):
        reply = "Paris" if "fp16" in profile["model"] else "Lyon"
        return ReplyModel(profile["model"], reply)

    result = run_quant_compare(
        {
            "baseline_pair": ("fp16", "llama-fp16"),
            "variants": [("q4", "llama-q4")],
            "provider": "ollama",
            "base_url": "http://127.0.0.1:11434",
            "preset": "quant_quick",
            "dataset": str(dataset),
            "tolerance": 0.05,
            "concurrency": 1,
            "out": str(out),
            "dry_run": False,
        },
        model_factory=factory,
    )
    assert result["verdict"] == "fail"
    markdown = (out / "report.md").read_text(encoding="utf-8")
    html = (out / "report.html").read_text(encoding="utf-8")
    saved = json.loads((out / "report.json").read_text(encoding="utf-8"))
    assert "Hallucination / factuality" in markdown
    assert "-1.0" in markdown
    assert "Output tokens/s" in html
    assert saved["baseline"] == "fp16"
    assert saved["variants"][0]["speed"]


def test_compare_view_includes_speed_and_power_deltas():
    left = {
        "run_id": "a",
        "created_at": "2026-10-13T15:00:00+00:00",
        "connection": {"model": "fp16"},
        "scorecard": {"categories": []},
        "edge_metrics": _edge(40.0, None),
    }
    right = {
        "run_id": "b",
        "created_at": "2026-10-13T16:00:00+00:00",
        "connection": {"model": "q4"},
        "scorecard": {"categories": []},
        "edge_metrics": _edge(55.0, None),
    }
    compared = compare_runs(right, left, [], [])
    by_name = {row["name"]: row for row in compared["edge"]}
    assert by_name["output_tokens_per_second"]["delta"] == 15.0
    assert by_name["peak_vram_mib"]["delta"] == "n/a"
    assert "NVIDIA" in by_name["peak_vram_mib"]["reason"]
    assert compare_runs({"run_id": "a", "scorecard": {}}, {"run_id": "b", "scorecard": {}}, [], [])["edge"] == []


def _load_standard_run(label: str) -> dict:
    path = Path(__file__).resolve().parents[1] / "fixtures" / "quant" / f"standard_{label}" / "run.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_standard_suite_run_json_feeds_the_delta_report():
    fp16 = _load_standard_run("fp16")
    q4 = _load_standard_run("q4")
    assert len(fp16["suites"]) + len(q4["suites"]) == 12
    assert fp16["scorecard"]["verdict"] == "No live categories scored"
    # Contradictory items must not replace the pass rate stored on run.json.
    ignored = [{"suite": "gsm8k", "passed": True, "counts_toward_score": False} for _ in range(20)]
    report = build_delta_report(
        [
            {"label": "fp16", "model": "llama-fp16", "run": fp16, "items": ignored},
            {"label": "q4", "model": "llama-q4", "run": q4, "items": []},
        ],
        baseline="fp16",
        tolerance=0.05,
    )
    rows = {row["name"]: row for row in report["variants"][0]["accuracy"]["benchmarks"]}
    assert set(rows) == {"gsm8k", "humaneval_plus", "ifeval", "math500", "mbpp_plus", "mmlu_pro"}
    assert rows["gsm8k"]["baseline"] == 0.8
    assert rows["gsm8k"]["variant"] == 0.75
    assert rows["gsm8k"]["baseline_passed"] == 16
    assert rows["gsm8k"]["variant_passed"] == 15
    assert rows["gsm8k"]["note"] == "within +/- 1 item"
    assert rows["gsm8k"]["status"] == "pass"
    assert rows["math500"]["delta"] == -1.0
    assert rows["math500"]["status"] == "fail"
    assert rows["math500"]["within_noise"] is False
    assert rows["mmlu_pro"]["label"] == "MMLU-Pro"
    assert report["verdict"] == "fail"
    text = render_markdown(report)
    html = render_html(report)
    benchmark = text.split("### Accuracy by benchmark", 1)[1].split("### Speed", 1)[0]
    assert "GSM8K" in benchmark
    assert "No rows." not in benchmark
    assert "No live categories scored" not in text
    assert "within +/- 1 item" in text
    assert "\u00b1" not in text
    assert "\u00b1" not in html
    assert "MATH-500" in html


def test_public_benchmark_items_count_when_the_suite_record_is_absent():
    report = build_delta_report(
        [
            _run("fp16", 1.0, items=[{"suite": "gsm8k", "category": "public_benchmark", "passed": True, "counts_toward_score": False}] * 4),
            _run("q4", 1.0, items=[{"suite": "gsm8k", "category": "public_benchmark", "passed": index < 3, "counts_toward_score": False} for index in range(4)]),
        ],
        baseline="fp16",
        tolerance=0.05,
    )
    row = report["variants"][0]["accuracy"]["benchmarks"][0]
    assert row["name"] == "gsm8k"
    assert row["variant_passed"] == 3
    assert row["delta"] == -0.25


def test_quant_quick_does_not_require_a_benchmark_directory():
    assert require_standard_data("quant_quick", None) is None


def test_fixture_directory_satisfies_the_standard_check():
    root = Path(__file__).resolve().parents[2] / "standard_bench" / "fixtures"
    found = require_standard_data("standard_capability_quick", str(root))
    assert Path(found) == root


def test_missing_standard_data_exits_before_the_run(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.delenv("LLM_EVAL_STANDARD_DATA", raising=False)
    monkeypatch.delenv("LLM_EVAL_STANDARD_FIXTURES", raising=False)

    class Boom:
        def __init__(self, *args, **kwargs):
            pass

        def start(self, **kwargs):
            raise AssertionError("run started without benchmark data")

        def items(self, run_id):
            return []

    monkeypatch.setattr("llm_eval_suite.runs.RunManager", Boom)
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "quant-compare",
                "--baseline",
                "fp16=llama",
                "--variant",
                "q4=llama-q4",
                "--preset",
                "standard_capability_quick",
                "--out",
                str(tmp_path / "quant"),
            ]
        )
    assert exc.value.code == 1
    assert "No standard benchmark directory" in capsys.readouterr().err


def test_failed_run_exits_nonzero(tmp_path: Path, monkeypatch, capsys):
    monkeypatch.setattr(
        "llm_eval.edge.cli.prepare_variant_vram",
        lambda *args, **kwargs: {
            "method": "card",
            "sample": "card",
            "baseline_vram_mib": None,
            "baseline_reason": "GPU did not report VRAM",
            "unloaded": [],
            "exclude_pids": [],
            "note": "",
        },
    )

    class Failed:
        def __init__(self, *args, **kwargs):
            pass

        def start(self, **kwargs):
            return {
                "run_id": "bad",
                "status": "failed",
                "error": "No standard benchmark directory. Set LLM_EVAL_STANDARD_DATA or pass --data.",
                "scorecard": {"verdict": "No live categories scored", "categories": []},
                "suites": [],
            }

        def items(self, run_id):
            return []

    monkeypatch.setattr("llm_eval_suite.runs.RunManager", Failed)
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "quant-compare",
                "--baseline",
                "fp16=llama",
                "--variant",
                "q4=llama-q4",
                "--out",
                str(tmp_path / "quant"),
            ]
        )
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert "Verdict: fail" in captured.out
    assert "No standard benchmark directory" in captured.err
    saved = json.loads((tmp_path / "quant" / "report.json").read_text(encoding="utf-8"))
    assert saved["verdict"] == "fail"


def test_pass_verdict_exits_zero(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "llm_eval.edge.cli.run_quant_compare",
        lambda spec, model_factory=None: {"verdict": "pass", "files": {}},
    )
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "quant-compare",
                "--baseline",
                "fp16=m",
                "--variant",
                "q4=m",
                "--out",
                str(tmp_path),
            ]
        )
    assert exc.value.code == 0


class _Body:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        if isinstance(self.payload, (dict, list)):
            return json.dumps(self.payload).encode("utf-8")
        return self.payload


def test_ollama_unload_records_a_card_baseline():
    posted = []

    def opener(request, timeout=0):
        if request.get_method() == "GET":
            return _Body({"models": [{"name": "llama3.2:3b-instruct-fp16"}]})
        posted.append(json.loads(request.data.decode("utf-8")))
        return _Body(b'{"done":true}\n')

    unloaded = unload_ollama_models("http://127.0.0.1:11434", opener=opener)
    assert unloaded == ["llama3.2:3b-instruct-fp16"]
    assert posted == [{"model": "llama3.2:3b-instruct-fp16", "keep_alive": 0}]
    policy = prepare_variant_vram(
        "ollama",
        "http://127.0.0.1:11434",
        gpu_reader=lambda: {"vram_mib": 512},
        unloader=lambda base_url: ["llama3.2:3b-instruct-fp16"],
        sleeper=lambda _seconds: None,
    )
    assert policy["method"] == "card peak after unloading other models"
    assert policy["baseline_vram_mib"] == 512
    assert "fp16" in policy["note"]
    refused = prepare_variant_vram(
        "ollama",
        "http://127.0.0.1:11434",
        gpu_reader=lambda: {"vram_mib": 100},
        unloader=lambda base_url: (_ for _ in ()).throw(OSError("connection refused")),
        sleeper=lambda _seconds: None,
    )
    assert refused["method"] == "card"
    assert "connection refused" in refused["note"]
    per_process = prepare_variant_vram(
        "vllm",
        "http://127.0.0.1:8000/v1",
        gpu_reader=lambda: {"vram_mib": 400},
        process_reader=lambda: [{"pid": "9", "name": "VLLM::Engine", "vram_mib": 8000}],
        sleeper=lambda _seconds: None,
    )
    assert per_process["method"] == "per-process"
    assert per_process["exclude_pids"] == ["9"]
    assert per_process["baseline_vram_mib"] == 400
    labeled = _run("fp16", 1.0)
    other = _run("q4", 1.0)
    labeled["run"]["edge_metrics"]["vram_method"] = {
        "value": "card peak after unloading other models",
        "reason": "Unloaded llama3.2:3b-instruct-fp16.",
    }
    labeled["run"]["edge_metrics"]["baseline_vram_mib"] = {"value": 512.0, "reason": ""}
    other["run"]["edge_metrics"]["vram_method"] = {
        "value": "card peak after unloading other models",
        "reason": "No other Ollama models were loaded.",
    }
    other["run"]["edge_metrics"]["baseline_vram_mib"] = {"value": 480.0, "reason": ""}
    other["run"]["edge_metrics"]["peak_vram_mib"] = {"value": 8547.0, "reason": ""}
    labeled["run"]["edge_metrics"]["peak_vram_mib"] = {"value": 11146.0, "reason": ""}
    compared = build_delta_report([labeled, other], baseline="fp16", tolerance=0.05)
    text = render_markdown(compared)
    assert "VRAM method (fp16): card peak after unloading other models (Unloaded llama3.2:3b-instruct-fp16.)." in text
    assert "Baseline VRAM" in text
    assert "8547" in text
