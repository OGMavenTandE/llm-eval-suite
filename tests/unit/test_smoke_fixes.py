"""Windows smoke fixes and the Results scorecard rules."""

from __future__ import annotations

import json
import sys
import time
import types
from pathlib import Path

import pytest
import yaml

from llm_eval.datasets.loader import load_dataset
from llm_eval.garak.live import attack_rates, count_report_attempts, parse_garak_report, run_garak, validity_from_items
from llm_eval.models.context import strip_think_blocks
from llm_eval.models.ollama_model import OllamaModel
from llm_eval.models.openai_model import OpenAIModel
from llm_eval_suite.compare import compare_runs
from llm_eval_suite.connections import ConnectionStore, build_model, judge_max_tokens, judge_timeout
from llm_eval_suite.council import judge_generate, template_narrative
from llm_eval_suite.runs import RunManager, factcheck_empty_validity, smooth_item_seconds
from llm_eval_suite.scoring import scorecard, withhold_category_scores
from llm_eval_suite.timing import TimingStore
from llm_eval_suite.suites import planned_suite_total, score_fact


def _fact_item(index: int, *, empty: bool) -> dict:
    return {
        "id": f"f{index}",
        "suite": "factcheck",
        "source": "live",
        "category": "hallucination_factuality",
        "passed": not empty,
        "response": "" if empty else "Paris",
        "empty": empty,
        "counts_toward_score": True,
        "score": 0 if empty else 1,
        "prompt": "Capital?",
        "expected": "Paris",
    }


def test_missing_garak_config_is_a_failed_live_scan(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("llm_eval.garak.live.garak_is_installed", lambda: True)
    seen = {}

    def fake_run(command, **kwargs):
        seen["config"] = command[command.index("--config") + 1]
        seen["cwd"] = kwargs.get("cwd")

        class Completed:
            returncode = 0
            stdout = "run config not found: garak-config-0.yaml"
            stderr = ""

        return Completed()

    work = tmp_path / "garak"
    result = run_garak(
        model_name="llama3.2:3b",
        base_url="http://127.0.0.1:11434",
        probes=["dan.Dan_11_0"],
        work_dir=work,
        runner=fake_run,
    )
    assert Path(seen["config"]).is_absolute()
    assert Path(seen["cwd"]).is_absolute()
    config = yaml.safe_load(Path(seen["config"]).read_text(encoding="utf-8"))
    generator = config["plugins"]["generators"]["openai"]["OpenAICompatible"]
    assert generator["uri"] == "http://127.0.0.1:11434/v1/"
    assert "generators" not in config
    assert result["source"] == "failed"
    assert result["label"] == "Failed live scan"
    assert result["validity"] == "invalid"
    assert "run config not found" in result["notes"]
    assert "fixture" not in result["notes"].lower()
    assert "exited without a usable report" not in result["notes"]


def test_scorecard_passes_only_when_every_live_category_clears_the_bar():
    weak = [
        {
            "category": "hallucination_factuality",
            "passed": index < 4,
            "source": "live",
            "counts_toward_score": True,
        }
        for index in range(10)
    ]
    strong = [
        {"category": "robustness", "passed": True, "source": "live", "counts_toward_score": True}
        for _ in range(5)
    ]
    card = scorecard(weak + strong)
    assert card["overall_pass_rate"] is None
    assert card["overall_pass_percent"] is None
    assert card["pass_bar"] == 0.8
    assert card["pass_bar_percent"] == 80.0
    assert card["meets_bar"] is False
    assert card["verdict"] == "1 of 2 live categories below the bar"
    assert card["failure_count"] == 6

    both = [
        {"category": "hallucination_factuality", "passed": True, "source": "live", "counts_toward_score": True}
        for _ in range(5)
    ]
    both += [
        {"category": "robustness", "passed": index < 4, "source": "live", "counts_toward_score": True}
        for index in range(5)
    ]
    passing = scorecard(both)
    assert passing["meets_bar"] is True
    assert passing["verdict"] == "Meets the 80% pass bar"

    mixed = [
        {"category": "security_jailbreak", "passed": False, "source": "fixture", "counts_toward_score": True},
        {"category": "hallucination_factuality", "passed": False, "source": "live", "counts_toward_score": True},
    ]
    counted = scorecard(mixed)
    assert counted["failure_count"] == 1
    security = counted["categories"][0]
    assert security["status"] == "fixture"
    assert security["pass_percent"] == 0.0


def test_template_uses_the_live_failure_count_not_the_sample_cap():
    items = [
        {
            "id": str(index),
            "passed": False,
            "source": "live",
            "counts_toward_score": True,
            "category": "hallucination_factuality",
            "prompt": f"q{index}",
            "response": "no",
            "score": 0,
        }
        for index in range(10)
    ]
    items.append(
        {
            "id": "fx",
            "passed": False,
            "source": "fixture",
            "counts_toward_score": True,
            "category": "security_jailbreak",
            "prompt": "fixture prompt",
            "response": "x",
            "score": 0,
        }
    )
    card = scorecard(items)
    text = template_narrative(
        {
            "run_id": "r",
            "preset": "quick",
            "connection": {"model": "stub"},
            "dataset": {"path": "sample.jsonl"},
            "scorecard": card,
        },
        items,
    )
    assert card["failure_count"] == 10
    assert "10 failing live prompts" in text
    assert "finished with 8" not in text


def test_empty_factcheck_run_is_invalid(tmp_path: Path):
    manager = RunManager(tmp_path / "runs")
    run_dir = manager.runs_dir / "emptyfact"
    manager._write_run(run_dir, {"run_id": "emptyfact", "status": "running", "connection": {"model": "qwen3:8b"}})
    items = [_fact_item(index, empty=True) for index in range(50)]
    suites = [{"name": "factcheck", "source": "live", "notes": "Scored with the correctness evaluator."}]

    class Model:
        device_name = "cuda"

    manager._finalize(run_dir, items, suites, "completed", error=None, model=Model())
    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert record["status"] == "invalid"
    assert record["validity"] == "invalid"
    assert record["validity_reason"] == "50 of 50 fact-check answers were empty"
    assert "INVALID fact-check." in record["suites"][0]["notes"]
    assert record["device"] == "cuda"
    assert record["scorecard"]["overall_pass_percent"] is None
    fact_row = next(row for row in record["scorecard"]["categories"] if row["category"] == "hallucination_factuality")
    assert fact_row["status"] == "withheld"
    assert fact_row["pass_percent"] is None
    assert factcheck_empty_validity(items)["validity"] == "invalid"

    few = [_fact_item(index, empty=index < 2) for index in range(10)]
    assert factcheck_empty_validity(few)["validity"] == "ok"
    mostly = [_fact_item(index, empty=index < 26) for index in range(50)]
    assert factcheck_empty_validity(mostly)["validity"] == "invalid"


def test_eta_uses_the_suite_total_and_clears_when_finished(tmp_path: Path):
    assert planned_suite_total("factcheck", {}, 50) == 50
    assert smooth_item_seconds(None, 2.0, fallback=10.0) == pytest.approx(7.6)
    manager = RunManager(tmp_path / "runs")
    run_dir = manager.runs_dir / "eta"
    manager._write_run(
        run_dir,
        {"run_id": "eta", "estimate": {"prompt_count": 175, "seconds_per_prompt": 1.0, "suite_rates": {}}},
    )
    unknown = {"name": "factcheck", "total": None, "done": 1, "status": "running"}
    manager._stamp_eta(run_dir, unknown, "eta")
    assert unknown["eta_seconds"] is None
    now = time.perf_counter()
    manager._eta_state[("eta", "garak")] = {"started": now - 1.0, "last": now - 0.4, "rate": None}
    early = {"name": "garak", "total": 10, "done": 1, "status": "running"}
    manager._stamp_eta(run_dir, early, "eta")
    assert early["eta_seconds"] is None
    assert early["eta_source"] == "estimating"
    early["done"] = 3
    manager._stamp_eta(run_dir, early, "eta")
    assert early["eta_seconds"] is not None
    assert early["eta_seconds"] < 175 * 60
    early["status"] = "completed"
    manager._stamp_eta(run_dir, early, "eta")
    assert early["eta_seconds"] is None

    matched = {"name": "garak", "total": 10, "done": 1, "status": "running"}
    manager._write_run(
        run_dir,
        {"run_id": "eta", "estimate": {"suite_rates": {"garak": 2.0}}},
    )
    manager._stamp_eta(run_dir, matched, "eta-matched")
    assert matched["eta_seconds"] == pytest.approx(18.0)
    assert matched["eta_source"] == "matched"


def test_judge_timeout_default_and_override(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("LLM_EVAL_JUDGE_TIMEOUT", raising=False)
    assert judge_timeout({}) == 90
    assert judge_timeout({"timeout": 15}) == 15
    monkeypatch.setenv("LLM_EVAL_JUDGE_TIMEOUT", "40")
    assert judge_timeout({}) == 40
    assert judge_timeout({"timeout": 12}) == 12
    saved = ConnectionStore(tmp_path).save_judges({"chairman": "a", "judges": [], "timeout": 120})
    assert saved["timeout"] == 120


def test_compare_surfaces_a_gain_among_zeros():
    left_items = [{"id": str(index), "score": 0.0, "prompt": "p"} for index in range(40)]
    right_items = [{"id": str(index), "score": 0.0, "prompt": "p"} for index in range(40)]
    right_items[-1]["score"] = 1.0
    compared = compare_runs(
        {"run_id": "earlier", "scorecard": {"categories": []}},
        {"run_id": "later", "scorecard": {"categories": []}},
        left_items,
        right_items,
    )
    assert compared["items"][0]["id"] == "39"
    assert compared["items"][0]["delta"] == 1.0


def test_number_words_and_expected_aliases(tmp_path: Path):
    _score, passed, mode = score_fact("How many planets?", "8", "There are Eight planets.")
    assert passed is True
    assert mode == "containment"
    _score, twenty, _mode = score_fact("How many?", "20", "Twenty ships sailed.")
    assert twenty is True
    _score, compound, _mode = score_fact("How many?", "156", "There were one hundred and fifty-six ships.")
    assert compound is True
    _score, water, _mode = score_fact("Formula?", "H2O", "The molecule is H₂O.")
    assert water is True
    _score, big, _mode = score_fact("How many?", "9999", "nine thousand nine hundred and ninety-nine")
    assert big is True
    assert strip_think_blocks("<think>hidden</think>Eight") == "Eight"
    assert strip_think_blocks("<think>never closed") == ""

    expected = tmp_path / "expected.jsonl"
    expected.write_text('{"prompt": "Capital?", "expected": "Paris"}\n', encoding="utf-8")
    assert load_dataset(str(expected))[0]["expected_answer"] == "Paris"
    answer = tmp_path / "answer.jsonl"
    answer.write_text('{"prompt": "Sum?", "answer": "4"}\n', encoding="utf-8")
    assert load_dataset(str(answer))[0]["expected_answer"] == "4"


def test_ollama_sends_think_false_and_retries_when_rejected(monkeypatch):
    model = build_model(
        {"type": "ollama", "model": "qwen3:8b", "base_url": "http://127.0.0.1:11434", "mode": "chat"}
    )
    assert isinstance(model, OpenAIModel)
    assert model.think is False

    payloads = []

    class Response:
        def __init__(self, status, body):
            self.status_code = status
            self._body = body
            self.text = json.dumps(body)

        def raise_for_status(self):
            if self.status_code >= 400:
                import requests

                raise requests.HTTPError(str(self.status_code))

        def json(self):
            return self._body

    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: A002
        payloads.append(dict(json))
        if "think" in json:
            return Response(400, {"error": {"message": "unknown field think"}})
        return Response(
            200,
            {"choices": [{"message": {"content": "<think>secret</think>Paris"}}], "usage": {}},
        )

    import requests

    monkeypatch.setattr(requests, "post", fake_post)
    assert model.generate("Capital?").text == "Paris"
    assert payloads[0]["think"] is False
    assert "think" not in payloads[1]

    native_payloads = []

    def native_post(url, json=None, timeout=None):  # noqa: A002
        native_payloads.append(dict(json))
        return Response(200, {"response": "<think>x</think>Paris", "eval_count": 1})

    monkeypatch.setattr(requests, "post", native_post)
    native = OllamaModel("qwen3:8b", {"base_url": "http://127.0.0.1:11434"})
    assert native.generate("Capital?").text == "Paris"
    assert native_payloads[0]["think"] is False


def test_hf_folder_uses_cuda_when_torch_can_see_it(tmp_path: Path, monkeypatch):
    folder = tmp_path / "weights"
    folder.mkdir()
    (folder / "config.json").write_text("{}", encoding="utf-8")
    calls = {}

    class FakeTensor:
        def __init__(self, data, device="cpu"):
            self.data = data
            self.device = device
            if data and isinstance(data[0], list):
                self.shape = (len(data), len(data[0]))
            else:
                self.shape = (len(data),)

        def __getitem__(self, item):
            if item == 0:
                return self
            if isinstance(item, slice):
                return FakeTensor([1, 2], device=self.device)
            raise IndexError(item)

    class FakeModel:
        def to(self, device, dtype=None):
            calls["to"] = (device, dtype)
            return self

        def eval(self):
            calls["eval"] = True

        def generate(self, **kwargs):
            calls["input_device"] = kwargs["input_ids"].device
            return FakeTensor([[1, 2, 3, 4]], device=kwargs["input_ids"].device)

    class FakeTokenizer:
        eos_token_id = 0
        chat_template = None

        def encode(self, text):
            return [1, 2, 3]

        def decode(self, tokens, skip_special_tokens=False):
            return "<think>hidden</think>Eight"

    def install(cuda: bool):
        torch_mod = types.ModuleType("torch")
        torch_mod.float16 = "float16"
        torch_mod.long = "long"
        torch_mod.cuda = types.SimpleNamespace(is_available=lambda: cuda)
        torch_mod.device = lambda name: name
        torch_mod.tensor = lambda data, dtype=None, device=None: FakeTensor(data, device or "cpu")
        torch_mod.ones_like = lambda tensor: FakeTensor(tensor.data, tensor.device)

        class NoGrad:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return False

        torch_mod.no_grad = NoGrad
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeTokenizer())
        transformers.AutoModelForCausalLM = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: FakeModel())
        monkeypatch.setitem(sys.modules, "torch", torch_mod)
        monkeypatch.setitem(sys.modules, "transformers", transformers)

    install(True)
    from llm_eval.models.hf_folder import HuggingFaceFolderModel

    cuda_model = HuggingFaceFolderModel("gpt2", {"folder": str(folder), "mode": "completions"})
    result = cuda_model.generate("How many?")
    assert cuda_model.device_name == "cuda"
    assert calls["to"] == ("cuda", "float16")
    assert calls["input_device"] == "cuda"
    assert result.text == "Eight"
    assert result.metadata["device"] == "cuda"

    install(False)
    calls.clear()
    cpu_model = HuggingFaceFolderModel("gpt2", {"folder": str(folder), "mode": "completions"})
    cpu_model.generate("How many?")
    assert cpu_model.device_name == "cpu"
    assert calls["to"] == ("cpu", None)
    assert calls["input_device"] == "cpu"


def test_report_download_favicon_and_run_list(tmp_path: Path):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from llm_eval_suite.app import create_app

    runs = tmp_path / "runs"
    app = create_app(data_dir=tmp_path / "data", runs_dir=runs, model_factory=lambda _profile: None)
    manager = app.state.runs

    def write(run_id: str, created_at: str, *, invalid: bool):
        record = {
            "run_id": run_id,
            "status": "invalid" if invalid else "completed",
            "validity": "invalid" if invalid else "ok",
            "created_at": created_at,
            "preset": "quick",
            "connection": {"model": "llama3.2:3b"},
            "scorecard": {
                "overall_pass_percent": 64.0,
                "pass_bar_percent": 80.0,
                "failure_count": 1,
                "verdict": "1 of 1 live categories below the bar",
                "categories": [],
            },
            "suites": [],
        }
        folder = runs / run_id
        folder.mkdir(parents=True)
        (folder / "run.json").write_text(json.dumps(record), encoding="utf-8")
        (folder / "items.jsonl").write_text(
            json.dumps(
                {
                    "passed": False,
                    "source": "live",
                    "category": "hallucination_factuality",
                    "prompt": "q",
                    "response": "",
                    "score": 0,
                }
            )
            + "\n",
            encoding="utf-8",
        )

    write("older", "2026-10-06T16:00:00+00:00", invalid=False)
    write("newer", "2026-10-06T18:00:00+00:00", invalid=True)
    listed = manager.list_runs()
    assert [row["run_id"] for row in listed] == ["newer", "older"]
    assert listed[0]["overall_pass_percent"] is None
    assert listed[1]["overall_pass_percent"] == 64.0

    client = TestClient(app)
    listing = client.get("/api/runs")
    assert listing.json()["pass_bar_percent"] == 80.0
    view = client.get("/api/runs/newer/report")
    assert view.status_code == 200
    assert "content-disposition" not in {key.lower() for key in view.headers}
    assert 'url("/static/fonts/' not in view.text
    assert "data:font/woff2;base64," in view.text
    downloaded = client.get("/api/runs/newer/report?download=1")
    disposition = downloaded.headers["content-disposition"]
    assert disposition.startswith("attachment;")
    assert 'filename="eval-report-newer.html"' in disposition
    icon = client.get("/favicon.ico")
    assert icon.status_code == 200
    assert icon.headers["content-type"].startswith("image/")
    page = client.get("/")
    assert page.status_code == 200
    assert 'rel="icon"' in page.text


def test_results_page_sources_and_browser_open_once(monkeypatch):
    app_js = Path("llm_eval_suite/static/app.js").read_text(encoding="utf-8")
    css = Path("llm_eval_suite/static/tokens.css").read_text(encoding="utf-8")
    html = Path("llm_eval_suite/static/index.html").read_text(encoding="utf-8")
    assert "PASS_BAR_PERCENT" not in app_js
    assert "Score withheld" in app_js
    assert "pass_bar_percent" in app_js
    assert "--ink-3: #687482" in css
    assert "break-all" not in css
    assert 'id="analysis-text"' in html and 'tabindex="0"' in html
    assert "nav-scroll-hint" in html
    assert ">1" in html and ">2" in html and ">3" in html

    from llm_eval_suite.app import open_app_browser

    monkeypatch.delenv("LLM_EVAL_BROWSER_OPENED", raising=False)
    opened = []
    monkeypatch.setattr("webbrowser.open", lambda url, new=0: opened.append(url) or True)
    if sys.platform == "win32":
        monkeypatch.setattr("os.startfile", lambda url: opened.append(url), raising=False)
    assert open_app_browser("http://127.0.0.1:8765") is True
    assert open_app_browser("http://127.0.0.1:8765") is False
    assert opened == ["http://127.0.0.1:8765"]


def test_duplicate_garak_attempts_score_once_and_agree(tmp_path: Path):
    path = Path(__file__).resolve().parents[1] / "fixtures" / "garak_duplicate_attempts.report.jsonl"
    raw_lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert count_report_attempts(path) == 5
    assert count_report_attempts(path) < len(raw_lines)
    items = parse_garak_report(path)
    by_id = {item["id"]: item for item in items}
    assert len(items) == len(by_id) == 4
    assert "garak:attempt-early" not in by_id
    bare = by_id["garak:attempt-nodet"]
    assert bare["passed"] is False
    assert bare["score"] is None
    assert bare["counts_toward_score"] is False
    dan = by_id["garak:attempt-dan"]
    assert dan["passed"] is False
    assert dan["score"] == pytest.approx(0.5)
    scored = [item for item in items if item.get("counts_toward_score")]
    assert len(scored) == 3
    card = scorecard(items)
    security = next(row for row in card["categories"] if row["category"] == "security_jailbreak")
    _attack, passed = attack_rates(items)
    assert passed == pytest.approx(0.5)
    assert security["pass_rate"] == pytest.approx(passed)
    assert security["pass_percent"] == pytest.approx(passed * 100, abs=0.05)
    assert security["pass_rate"] != pytest.approx(1 / 3, abs=0.01)

    empty = tmp_path / "empty.report.jsonl"
    lines = []
    for seq in range(3):
        base = {
            "entry_type": "attempt",
            "uuid": f"empty-{seq}",
            "seq": seq,
            "probe_classname": "leakreplay.LiteratureCloze",
            "prompt": f"prompt {seq}",
            "outputs": [""],
        }
        lines.append(dict(base, status=1))
        lines.append(dict(base, status=2, detector_results={"leakreplay.Detector": [0.0]}))
    empty.write_text("".join(json.dumps(row) + "\n" for row in lines), encoding="utf-8")
    assert count_report_attempts(empty) == 3
    parsed = parse_garak_report(empty)
    assert len(parsed) == 3
    validity = validity_from_items(parsed)
    assert validity["validity"] == "invalid"
    assert "3 of 3" in validity["reason"]
    assert "6 of 6" not in validity["reason"]


def test_timing_rate_stays_on_the_same_model_and_suite(tmp_path: Path):
    store = TimingStore(tmp_path / "timing.json")
    qwen = {"type": "ollama", "base_url": "http://127.0.0.1:11434", "model": "qwen2.5"}
    other = {"type": "ollama", "base_url": "http://127.0.0.1:11434", "model": "qwen3"}
    store.record(10, 5, connection=qwen, suite="garak")
    assert store.rate_for(qwen, "garak") == pytest.approx(2.0)
    assert store.rate_for(other, "garak") is None
    assert store.rate_for(qwen, "factcheck") is None
    legacy = tmp_path / "legacy.json"
    legacy.write_text(
        json.dumps({"seconds_per_prompt": 1.0, "source": "measured", "prompts": 10}),
        encoding="utf-8",
    )
    assert TimingStore(legacy).rate_for(qwen, "garak") is None


def test_invalid_run_withholds_category_scores_and_compare_deltas():
    card = scorecard(
        [
            {
                "id": "s1",
                "suite": "garak",
                "category": "security_jailbreak",
                "source": "live",
                "passed": True,
                "score": 1.0,
                "counts_toward_score": True,
                "response": "No",
            }
        ]
    )
    withheld = withhold_category_scores(card)
    security = next(row for row in withheld["categories"] if row["category"] == "security_jailbreak")
    assert security["status"] == "withheld"
    assert security["pass_percent"] is None
    assert security["pass_rate"] is None
    assert withheld["verdict"] == "Score withheld"
    not_run = next(row for row in withheld["categories"] if row["category"] == "retrieval")
    assert not_run["status"] == "not_run"

    compared = compare_runs(
        {"run_id": "bad", "status": "invalid", "validity": "invalid", "scorecard": card},
        {"run_id": "ok", "status": "completed", "validity": "ok", "scorecard": card},
        [{"id": "s1", "score": 0.0, "prompt": "p"}],
        [{"id": "s1", "score": 1.0, "prompt": "p"}],
    )
    assert compared["left_invalid"] is True
    assert compared["right_invalid"] is False
    assert compared["items"] == []
    assert all(row["delta"] is None for row in compared["categories"])
    assert all(row["left_pass_rate"] is None for row in compared["categories"])


def test_judge_generate_disables_think_and_caps_tokens(monkeypatch):
    monkeypatch.delenv("LLM_EVAL_JUDGE_MAX_TOKENS", raising=False)
    assert judge_max_tokens({}) == 1200
    assert judge_max_tokens({"max_tokens": 800}) == 800
    seen = {}

    class Fake:
        def generate(self, prompt, max_tokens=None):
            seen["max_tokens"] = max_tokens
            seen["prompt"] = prompt

            class Result:
                text = "<think>hidden</think>The jailbreak probes failed."

            return Result()

    def fake_build(profile):
        seen["think"] = profile.get("think")
        return Fake()

    monkeypatch.setattr("llm_eval_suite.connections.build_model", fake_build)
    generate = judge_generate(
        {"type": "openai", "model": "qwen3", "base_url": "http://127.0.0.1:11434"},
        max_tokens=1200,
    )
    assert seen["think"] is False
    assert generate("Summarize the run.") == "The jailbreak probes failed."
    assert seen["max_tokens"] == 1200
