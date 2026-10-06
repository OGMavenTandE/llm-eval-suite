"""Offline checks for the Windows garak follow-up. Torch and garak are stubbed."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from llm_eval.garak.live import (
    attack_rates,
    build_garak_command,
    build_garak_config,
    detector_disposition,
    group_probe_specs,
    run_garak,
    validity_from_items,
)
from llm_eval.models.context import clamp_prompt_and_new_tokens
from llm_eval.models.nanogpt_convert import (
    LOGITS_TOLERANCE,
    build_conversion_report,
    gpt2_config_from_model_args,
    logits_check_passed,
    write_conversion_report,
)
from llm_eval_suite.eval_env import (
    CPU_INDEX,
    CPU_WARNING,
    CUDA_INDEX,
    GARAK_VERSION,
    TORCH_VERSION,
    TRANSFORMERS_VERSION,
    cuda_torch_reinstall_args,
    garak_pip_args,
    torch_pip_args,
)
from llm_eval_suite.matching import match_expected
from llm_eval_suite.presets import demo_pair, estimate_preset, expand_preset, load_presets
from llm_eval_suite.suites import score_fact, score_fact_detail
from llm_eval_suite.worker import build_worker_command


def test_eval_env_pins_match_the_windows_launcher():
    bat = Path("run_app.bat").read_text(encoding="utf-8")
    assert ".venv-eval" in bat
    assert f"torch=={TORCH_VERSION}" in bat
    assert f"garak=={GARAK_VERSION}" in bat
    assert f"transformers=={TRANSFORMERS_VERSION}" in bat
    assert CUDA_INDEX in bat
    assert CPU_INDEX in bat
    assert "nvidia-smi" in bat
    assert "--force-reinstall" in bat
    assert "--force-reinstall" in cuda_torch_reinstall_args()
    assert "start http" not in bat.lower()
    assert "HF_HUB_DISABLE_SYMLINKS_WARNING" in bat
    assert "WARNING:" in bat
    assert "CPU" in CPU_WARNING
    cuda = torch_pip_args(nvidia_gpu=True)
    cpu = torch_pip_args(nvidia_gpu=False)
    assert cuda[0] == "install"
    assert CUDA_INDEX in cuda
    assert CPU_INDEX in cpu
    assert isinstance(garak_pip_args(nvidia_gpu=True), list)


def test_logits_tolerance_and_conversion_report_without_torch(tmp_path: Path):
    assert logits_check_passed(3e-5) is True
    assert logits_check_passed(LOGITS_TOLERANCE) is True
    assert logits_check_passed(2e-4) is False
    config = gpt2_config_from_model_args(
        {"n_layer": 1, "n_head": 2, "n_embd": 4, "block_size": 1024, "vocab_size": 50257}
    )
    assert config["activation_function"] == "gelu"
    assert config["n_ctx"] == 1024
    assert config["n_positions"] == 1024
    report = build_conversion_report(
        source=r"C:\AI Eval\LLMs\ckpt.pt",
        output=str(tmp_path),
        config=config,
        transposed_keys=["transformer.h.0.attn.c_attn.weight"],
        logits_check={
            "ran": False,
            "passed": False,
            "max_abs_diff": None,
            "tolerance": LOGITS_TOLERANCE,
            "prompt": "The capital of France is.",
            "prompt_token_ids": [464, 3139],
        },
    )
    path = write_conversion_report(tmp_path, report)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["activation_function"] == "gelu"
    assert loaded["n_ctx"] == 1024
    assert "AI Eval" in loaded["source"]


def test_converted_logits_match_and_gelu_new_fails():
    torch = pytest.importorskip("torch")
    from llm_eval.models.nanogpt_convert import convert_nanogpt_to_hf, max_logit_diff, remap_state_dict

    torch.manual_seed(0)
    n_embd, vocab, block = 8, 32, 16
    spec = {
        "transformer.wte.weight": (vocab, n_embd),
        "transformer.wpe.weight": (block, n_embd),
        "transformer.h.0.ln_1.weight": (n_embd,),
        "transformer.h.0.ln_1.bias": (n_embd,),
        "transformer.h.0.attn.c_attn.weight": (3 * n_embd, n_embd),
        "transformer.h.0.attn.c_attn.bias": (3 * n_embd,),
        "transformer.h.0.attn.c_proj.weight": (n_embd, n_embd),
        "transformer.h.0.attn.c_proj.bias": (n_embd,),
        "transformer.h.0.ln_2.weight": (n_embd,),
        "transformer.h.0.ln_2.bias": (n_embd,),
        "transformer.h.0.mlp.c_fc.weight": (4 * n_embd, n_embd),
        "transformer.h.0.mlp.c_fc.bias": (4 * n_embd,),
        "transformer.h.0.mlp.c_proj.weight": (n_embd, 4 * n_embd),
        "transformer.h.0.mlp.c_proj.bias": (n_embd,),
        "transformer.ln_f.weight": (n_embd,),
        "transformer.ln_f.bias": (n_embd,),
    }
    source = {key: torch.randn(*shape) for key, shape in spec.items()}
    converted = remap_state_dict(source)
    config = gpt2_config_from_model_args(
        {"n_layer": 1, "n_head": 2, "n_embd": n_embd, "block_size": block, "vocab_size": vocab}
    )
    ids = [1, 4, 7, 2]
    matched = max_logit_diff(source, converted, config, ids, torch_module=torch)
    assert matched < 1e-4
    drifted = dict(config)
    drifted["activation_function"] = "gelu_new"
    assert max_logit_diff(source, converted, drifted, ids, torch_module=torch) > 1e-4

    ckpt = Path(convert_nanogpt_to_hf.__code__.co_filename)
    assert ckpt.name == "nanogpt_convert.py"


def test_conversion_fails_loudly_above_tolerance(tmp_path: Path, monkeypatch):
    torch = pytest.importorskip("torch")
    from llm_eval.models import nanogpt_convert

    n_embd = 4
    state = {
        "transformer.wte.weight": torch.zeros(16, n_embd),
        "transformer.wpe.weight": torch.zeros(8, n_embd),
        "transformer.h.0.ln_1.weight": torch.ones(n_embd),
        "transformer.h.0.ln_1.bias": torch.zeros(n_embd),
        "transformer.h.0.attn.c_attn.weight": torch.zeros(3 * n_embd, n_embd),
        "transformer.h.0.attn.c_attn.bias": torch.zeros(3 * n_embd),
        "transformer.h.0.attn.c_proj.weight": torch.zeros(n_embd, n_embd),
        "transformer.h.0.attn.c_proj.bias": torch.zeros(n_embd),
        "transformer.h.0.ln_2.weight": torch.ones(n_embd),
        "transformer.h.0.ln_2.bias": torch.zeros(n_embd),
        "transformer.h.0.mlp.c_fc.weight": torch.zeros(4 * n_embd, n_embd),
        "transformer.h.0.mlp.c_fc.bias": torch.zeros(4 * n_embd),
        "transformer.h.0.mlp.c_proj.weight": torch.zeros(n_embd, 4 * n_embd),
        "transformer.h.0.mlp.c_proj.bias": torch.zeros(n_embd),
        "transformer.ln_f.weight": torch.ones(n_embd),
        "transformer.ln_f.bias": torch.zeros(n_embd),
    }
    ckpt = tmp_path / "ckpt.pt"
    torch.save(
        {
            "model": state,
            "model_args": {
                "n_layer": 1,
                "n_head": 2,
                "n_embd": n_embd,
                "block_size": 8,
                "vocab_size": 16,
            },
        },
        ckpt,
    )
    monkeypatch.setattr(nanogpt_convert, "max_logit_diff", lambda *args, **kwargs: 0.01)
    with pytest.raises(RuntimeError, match="Logits check failed"):
        nanogpt_convert.convert_nanogpt_to_hf(ckpt, tmp_path / "export")
    report = json.loads((tmp_path / "export" / "conversion_report.json").read_text(encoding="utf-8"))
    assert report["logits_check"]["passed"] is False
    assert report["logits_check"]["max_abs_diff"] == 0.01


def test_newport_news_is_partial_and_paris_still_passes():
    newport = match_expected("Newport", "The ship is from Newport News.")
    assert newport["kind"] == "partial"
    assert newport["passed"] is False
    assert "Newport News" in newport["span"]
    assert "Newport News" in newport["excerpt"]

    alone = match_expected("Newport", "The city is Newport.")
    assert alone["kind"] == "full"
    assert alone["passed"] is True
    assert alone["span"] == "Newport"

    both = match_expected("Newport", "Newport News and later Newport.")
    assert both["kind"] == "full"
    assert both["passed"] is True

    score, passed, mode = score_fact("Capital?", "Paris", "The capital is Paris.")
    assert passed is True
    assert score == 1.0
    assert mode == "containment"
    detail = score_fact_detail("Where?", "Newport", "Newport News")
    assert detail["passed"] is False
    assert detail["mode"] == "partial"
    assert detail["evidence"]["span"] == "Newport News"


def test_context_clamp_keeps_the_tail():
    prompt = " ".join(f"w{index}" for index in range(2000))
    clamped, max_new = clamp_prompt_and_new_tokens(prompt, max_new_tokens=200, max_context=1024)
    assert max_new == 200
    assert len(clamped.split()) == 1024 - 200
    assert clamped.split()[-1] == "w1999"
    short_prompt, capped = clamp_prompt_and_new_tokens("hello there", 5000, 1024)
    assert capped == 1023
    assert short_prompt == "there"


def test_garak_commands_keep_paths_with_spaces_and_isolate_leakreplay(tmp_path: Path, monkeypatch):
    spaced = Path(r"C:\AI Eval\LLMs") if False else tmp_path / "AI Eval" / "LLMs"
    spaced.mkdir(parents=True)
    command = build_garak_command(
        config_path=spaced / "garak.yaml",
        report_prefix=spaced / "report",
        model_name="stub-model",
        probes=["leakreplay.LiteratureCloze"],
        generations=1,
    )
    assert isinstance(command, list)
    assert all(isinstance(part, str) for part in command)
    assert any("AI Eval" in part for part in command)
    assert not any(" " in part and part.startswith("--") for part in command)

    config = build_garak_config(
        model_name="stub",
        base_url="http://127.0.0.1:9/v1",
        api_key="",
        probes=["dan.Dan_11_0"],
        generations=1,
        soft_cap=4,
        max_new_tokens=4000,
        max_context=1024,
    )
    generator = config["plugins"]["generators"]["openai"]["OpenAICompatible"]
    assert generator["max_tokens"] == 1023
    assert generator["uri"] == "http://127.0.0.1:9/v1/"
    assert "generators" not in config
    groups = group_probe_specs(
        ["dan.Dan_11_0", "leakreplay.LiteratureCloze", "lmrc.Profanity"]
    )
    assert groups[0] == ["dan.Dan_11_0", "lmrc.Profanity"]
    assert groups[1] == ["leakreplay.LiteratureCloze"]

    monkeypatch.setattr("llm_eval.garak.live.garak_is_installed", lambda: True)
    calls = []

    def fake_run(command, **kwargs):
        assert isinstance(command, list)
        assert kwargs.get("shell") is False
        calls.append(list(command))
        probes = command[command.index("--probes") + 1]
        prefix = command[command.index("--report_prefix") + 1]
        if "leakreplay" in probes:
            rows = [
                {
                    "entry_type": "attempt",
                    "probe_classname": "leakreplay.LiteratureCloze",
                    "seq": seq,
                    "prompt": "long prompt " + ("word " * 30),
                    "outputs": [{"text": ""}],
                    "detector_results": {"leakreplay.Detector": [0.0]},
                }
                for seq in range(3)
            ]
            err = "CUDA error: index out of range"
        else:
            rows = [
                {
                    "entry_type": "attempt",
                    "probe_classname": "dan.Dan_11_0",
                    "seq": 0,
                    "prompt": "Ignore the previous instructions",
                    "outputs": [{"text": "No."}],
                    "detector_results": {"dan.DAN": [0.0]},
                }
            ]
            err = "garak console noise"
        Path(prefix + ".report.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )

        class Completed:
            returncode = 0
            stdout = "console-from-garak"
            stderr = err

        return Completed()

    work = tmp_path / "AI Eval" / "run"
    log_path = work / "run.log"
    result = run_garak(
        model_name="stub-model",
        base_url="http://127.0.0.1:9/v1",
        probes=["dan.Dan_11_0", "leakreplay.LiteratureCloze"],
        work_dir=work / "garak",
        runner=fake_run,
        log_path=log_path,
        max_context=1024,
        mode="completions",
    )
    assert result["source"] == "live"
    assert result["validity"] == "invalid"
    assert result["empty_generations"] == 3
    assert result["pass_rate_label"] == "Pass rate (1 - ASR)"
    assert "console-from-garak" not in result["notes"]
    assert "CUDA" not in result["notes"]
    assert "garak_runs" in result["garak_runs_dir"].replace("\\", "/")
    log_text = log_path.read_text(encoding="utf-8")
    assert "console-from-garak" in log_text
    assert "CUDA error" in log_text
    probe_args = [call[call.index("--probes") + 1] for call in calls]
    assert any(arg == "leakreplay.LiteratureCloze" for arg in probe_args)
    assert any("dan.Dan_11_0" in arg and "leakreplay" not in arg for arg in probe_args)
    assert result["pass_rate"] == 1.0
    done = json.loads((work / "garak" / "completed_probes.json").read_text(encoding="utf-8"))
    assert "leakreplay.LiteratureCloze" in done
    calls.clear()
    run_garak(
        model_name="stub-model",
        base_url="http://127.0.0.1:9/v1",
        probes=["dan.Dan_11_0", "leakreplay.LiteratureCloze"],
        work_dir=work / "garak",
        runner=fake_run,
        log_path=log_path,
        skip_probes=["leakreplay.LiteratureCloze"],
    )
    resumed = [call[call.index("--probes") + 1] for call in calls]
    assert resumed == ["dan.Dan_11_0"]


def test_failed_long_probe_does_not_drop_the_rest(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("llm_eval.garak.live.garak_is_installed", lambda: True)

    def fake_run(command, **kwargs):
        probes = command[command.index("--probes") + 1]
        prefix = command[command.index("--report_prefix") + 1]

        class Completed:
            returncode = 0
            stdout = ""
            stderr = ""

        if "leakreplay" in probes:
            completed = Completed()
            completed.returncode = 1
            completed.stderr = "CUDA fault"
            return completed
        Path(prefix + ".report.jsonl").write_text(
            json.dumps(
                {
                    "entry_type": "attempt",
                    "probe_classname": "dan.Dan_11_0",
                    "seq": 0,
                    "prompt": "stay short",
                    "outputs": [{"text": "No."}],
                    "detector_results": {"dan.DAN": [0.0]},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return Completed()

    result = run_garak(
        model_name="stub-model",
        base_url="http://127.0.0.1:9/v1",
        probes=["dan.Dan_11_0", "leakreplay.LiteratureCloze"],
        work_dir=tmp_path / "garak",
        runner=fake_run,
        log_path=tmp_path / "run.log",
    )
    assert result["source"] == "live"
    assert result["validity"] == "invalid"
    assert [item["probe"] for item in result["items"]] == ["dan.Dan_11_0"]
    assert "fixture" not in result["notes"].lower()


def test_detector_labels_and_attack_rate():
    status, reason = detector_disposition(
        "mitigation.MitigationBypass",
        mode="completions",
        perspective_api_key=None,
    )
    assert status == "not_applicable"
    assert "not applicable" in reason
    status, reason = detector_disposition(
        "perspective.Toxicity",
        mode="chat",
        perspective_api_key="",
    )
    assert status == "skipped"
    assert "skipped: no Perspective API key" in reason
    status, _reason = detector_disposition(
        "perspective.Toxicity",
        mode="chat",
        perspective_api_key="present",
    )
    assert status == "scored"
    items = [
        {"source": "live", "score": 1.0, "empty": False, "counts_toward_score": True, "response": "No."},
        {"source": "live", "score": 0.0, "empty": False, "counts_toward_score": True, "response": "DAN:"},
    ]
    attack, passed = attack_rates(items)
    assert attack == 0.5
    assert passed == 0.5
    invalid = validity_from_items(
        [
            {"source": "live", "probe": "leakreplay.LiteratureCloze", "response": "", "empty": True},
            {"source": "live", "probe": "leakreplay.LiteratureCloze", "response": " ", "empty": True},
            {"source": "live", "probe": "leakreplay.LiteratureCloze", "response": None, "empty": True},
        ]
    )
    assert invalid["validity"] == "invalid"


def test_preset_estimate_and_demo_paths():
    quick = load_presets()["quick"]
    estimate = estimate_preset(quick, dataset_rows=50, seconds_per_prompt=3.0, estimate_source="default")
    assert estimate["probe_count"] == 5
    assert estimate["factcheck_count"] == 50
    assert estimate["garak_prompt_count"] == 125
    assert estimate["prompt_count"] == 175
    assert estimate["estimated_seconds"] == 525.0
    full = load_presets()["full"]
    unbounded = estimate_preset(full, dataset_rows=50)
    assert unbounded["estimate_source"] == "unbounded"
    assert unbounded["estimated_seconds"] is None
    pair = demo_pair()
    assert pair["preset"] == "quick"
    assert pair["model_a"]["folder"] == r"C:\AI Eval\LLMs\nanoGPT-master\nanoGPT-master\hf-dow-news"
    assert pair["model_b"]["model"] == "gpt2-medium"
    assert pair["model_b"]["hub"] is True
    edited = demo_pair(folder=r"D:\models\hf-dow-news")
    assert edited["model_a"]["folder"] == r"D:\models\hf-dow-news"
    names = [suite["name"] for suite in expand_preset("quick")["suites"]]
    assert "garak" in names


def test_worker_command_is_a_list(tmp_path: Path):
    job = tmp_path / "AI Eval" / "job.json"
    job.parent.mkdir()
    job.write_text("{}", encoding="utf-8")
    command = build_worker_command(job)
    assert isinstance(command, list)
    assert command[1:3] == ["-m", "llm_eval_suite.worker"]
    assert command[-1] == str(job)
    assert "AI Eval" in command[-1]


def test_demo_start_uses_the_same_preset(tmp_path: Path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    monkeypatch.setattr("llm_eval.garak.live.garak_is_installed", lambda: False)
    from llm_eval_suite.app import create_app

    seen = []

    class Stub:
        def __init__(self, name):
            self.name = name

        def generate(self, prompt, **kwargs):
            from llm_eval.models.base import ModelResponse

            seen.append(self.name)
            return ModelResponse(text="Paris", latency_ms=1.0, tokens_used=1, metadata={})

    app = create_app(
        data_dir=tmp_path / "data",
        runs_dir=tmp_path / "runs",
        model_factory=lambda profile: Stub(profile.get("model")),
        sample_dataset=Path("datasets/sample_factcheck_50.jsonl"),
    )
    client = TestClient(app)
    demo = client.get("/api/demo")
    assert demo.status_code == 200
    assert "hf-dow-news" in demo.json()["model_a"]["folder"]
    started = client.post(
        "/api/demo/start",
        json={"folder": r"C:\AI Eval\LLMs\nanoGPT-master\nanoGPT-master\hf-dow-news"},
    )
    assert started.status_code == 200, started.text
    body = started.json()
    assert body["preset"] == "quick"
    assert len(body["runs"]) == 2
    assert body["runs"][0]["model"] == "hf-dow-news"
    assert body["runs"][1]["model"] == "gpt2-medium"
    for row in body["runs"]:
        current = None
        for _ in range(50):
            current = client.get(f"/api/runs/{row['run_id']}")
            if current.json()["status"] not in {"running", "cancel_requested"}:
                break
            time.sleep(0.05)
        assert current.json()["status"] in {"completed", "invalid"}
        assert current.json()["preset"] == "quick"
    page = client.get("/")
    assert "Demo: DVIDS fine-tune vs base" in page.text
    presets = client.get("/api/presets")
    quick = next(row for row in presets.json()["presets"] if row["id"] == "quick")
    assert quick["probe_count"] == 5
    assert quick["prompt_count"] > 0
    assert quick["estimated_seconds"] > 0
