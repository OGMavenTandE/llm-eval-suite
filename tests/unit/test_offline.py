"""Offline mode: no hub download, no OpenAI cloud call, no outbound socket on a stub run."""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path

import pytest

from llm_eval.offline import (
    AIRGAP_BUILD,
    DETECTOR_PINS,
    OFFLINE_ENV,
    activate_run_offline,
    apply_offline_environment,
    demo_base_folder,
    demo_base_warning,
    detector_skip_notes,
    garak_detector_config,
    hub_allowed,
    hub_refused_message,
    install_import_guard,
    offline_active,
    resolve_pretrained_source,
)


def test_env_switch_sets_hub_offline_before_any_hub_import(monkeypatch):
    monkeypatch.delenv("LLM_EVAL_OFFLINE", raising=False)
    for key in OFFLINE_ENV:
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LLM_EVAL_OFFLINE", "1")
    assert "import transformers" not in Path("llm_eval/offline.py").read_text(encoding="utf-8")
    assert "import huggingface_hub" not in Path("llm_eval/offline.py").read_text(encoding="utf-8")
    assert "import garak" not in Path("llm_eval/offline.py").read_text(encoding="utf-8")
    apply_offline_environment()
    install_import_guard()
    for key, value in OFFLINE_ENV.items():
        assert os.environ[key] == value
    assert offline_active() is True
    assert hub_allowed() is False


def test_startup_hook_precedes_hub_imports():
    app = Path("llm_eval_suite/app.py").read_text(encoding="utf-8")
    assert app.index("apply_startup_offline()") < app.index("import requests")
    assert app.index("apply_startup_offline()") < app.index("from llm_eval_suite.connections")
    folder = Path("llm_eval/models/hf_folder.py").read_text(encoding="utf-8")
    assert folder.index("apply_startup_offline()") < folder.index("from transformers import")
    worker = Path("llm_eval_suite/worker.py").read_text(encoding="utf-8")
    assert worker.index("activate_run_offline") < worker.index("from llm_eval_suite.runs import RunManager")


def test_settings_flag_and_config_flag(tmp_path: Path, monkeypatch):
    monkeypatch.delenv("LLM_EVAL_OFFLINE", raising=False)
    from llm_eval.offline import read_settings, write_settings_offline

    assert offline_active() is False
    status = write_settings_offline(tmp_path, True)
    assert status["offline"] is True
    assert status["openai_cloud_blocked"] is True
    assert "OpenAI cloud API" in status["message"]
    assert read_settings(tmp_path)["offline"] is True
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert os.environ["HF_HUB_DISABLE_TELEMETRY"] == "1"
    write_settings_offline(tmp_path, False)
    assert offline_active() is False

    activate_run_offline(config={"offline": True})
    assert offline_active() is True
    assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
    assert os.environ["HF_DATASETS_OFFLINE"] == "1"


def test_preset_offline_field_blocks_openai_cloud(monkeypatch):
    monkeypatch.delenv("LLM_EVAL_OFFLINE", raising=False)
    from llm_eval.models.openai_model import OpenAIModel
    from llm_eval_suite.presets import expand_preset

    quick = expand_preset("quick")
    assert quick["offline"] is False
    gov = expand_preset("government_te")
    assert gov["offline"] is True
    assert "OpenAI cloud API" in gov["warning"]
    activate_run_offline(preset=gov)
    model = OpenAIModel("gpt-4o-mini", {})
    with pytest.raises(RuntimeError, match="OpenAI cloud API"):
        model.generate("hello")


def test_local_openai_endpoint_still_runs_when_offline(monkeypatch):
    monkeypatch.setenv("LLM_EVAL_OFFLINE", "1")
    activate_run_offline()
    from llm_eval.models.openai_model import OpenAIModel

    calls = []

    class Response:
        status_code = 200
        text = ""

        def raise_for_status(self):
            return None

        def json(self):
            return {"choices": [{"text": "pong"}], "usage": {"total_tokens": 1}}

    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: A002
        calls.append(url)
        return Response()

    import requests

    monkeypatch.setattr(requests, "post", fake_post)
    model = OpenAIModel("local", {"base_url": "http://127.0.0.1:9/v1", "api_key": "", "mode": "completions"})
    assert model.generate("ping").text == "pong"
    assert calls == ["http://127.0.0.1:9/v1/completions"]


def test_offline_refuses_hub_id_and_names_it(monkeypatch):
    monkeypatch.setenv("LLM_EVAL_OFFLINE", "1")
    activate_run_offline()
    from llm_eval.models.hf_folder import HuggingFaceFolderModel

    model = HuggingFaceFolderModel("gpt2-medium", {"folder": "gpt2-medium", "hub": True})
    with pytest.raises(RuntimeError, match="gpt2-medium") as caught:
        model._load()
    assert "Nothing was downloaded" in str(caught.value)
    _local, error = resolve_pretrained_source("gpt2-medium")
    assert error is not None
    assert "gpt2-medium" in error


def test_airgap_build_disables_hub_branch_without_offline_env(monkeypatch):
    monkeypatch.delenv("LLM_EVAL_OFFLINE", raising=False)
    monkeypatch.delenv("LLM_EVAL_AIRGAP_BUILD", raising=False)
    assert AIRGAP_BUILD is False
    monkeypatch.setattr("llm_eval.offline.AIRGAP_BUILD", True)
    monkeypatch.setattr("llm_eval.models.hf_folder.hub_allowed", lambda: False)
    from llm_eval.models.hf_folder import HuggingFaceFolderModel
    from llm_eval import offline

    assert offline.airgap_build() is True
    assert offline.hub_allowed() is False
    message = hub_refused_message("org/model")
    assert "LLM_EVAL_AIRGAP_BUILD" in message
    assert "org/model" in message
    model = HuggingFaceFolderModel("org/model", {"folder": "org/model", "hub": True})
    with pytest.raises(RuntimeError, match="LLM_EVAL_AIRGAP_BUILD"):
        model._load()

    monkeypatch.setattr("llm_eval.offline.AIRGAP_BUILD", False)
    monkeypatch.setenv("LLM_EVAL_AIRGAP_BUILD", "1")
    assert offline.airgap_build() is True
    assert offline.hub_allowed() is False


def test_demo_preset_points_at_a_local_folder(monkeypatch):
    from llm_eval_suite.presets import demo_pair, list_presets

    pair = demo_pair()
    assert pair["model_b"]["hub"] is False
    assert pair["model_b"]["folder"] == "models/gpt2-medium"
    assert "models/gpt2-medium" in demo_base_warning()
    listed = next(row for row in list_presets() if row["id"] == "demo_dvids")
    assert listed["offline"] is True
    assert "models/gpt2-medium" in listed["warning"]
    assert "does not download" in listed["warning"]
    monkeypatch.setenv("LLM_EVAL_DEMO_BASE_FOLDER", r"D:\weights\gpt2-medium")
    assert demo_base_folder("models/gpt2-medium") == r"D:\weights\gpt2-medium"
    moved = demo_pair()
    assert moved["model_b"]["folder"] == r"D:\weights\gpt2-medium"
    assert r"D:\weights\gpt2-medium" in next(row for row in list_presets() if row["id"] == "demo_dvids")["warning"]


def test_garak_detectors_pin_to_local_dirs_and_skip_when_missing(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LLM_EVAL_OFFLINE", "1")
    activate_run_offline()
    notes = detector_skip_notes("misleading")
    assert len(notes) == 2
    blob = " ".join(notes)
    for pin in DETECTOR_PINS:
        assert pin.repo_id in blob
        assert pin.revision in blob
        for name in pin.files:
            assert name in blob
        assert "Nothing was downloaded" in blob
    assert detector_skip_notes(["dan.Dan_11_0"]) == []
    config = garak_detector_config()
    refute = config["misleading"]["MustRefuteClaimModel"]
    nli = config["misleading"]["MustContradictNLI"]
    assert refute["skip"] is True
    assert nli["skip"] is True
    assert "garak-llm" not in json.dumps(config)

    ready = tmp_path / "refutation"
    ready.mkdir()
    (ready / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setenv("LLM_EVAL_GARAK_REFUTATION_DIR", str(ready))
    pinned = garak_detector_config()["misleading"]["MustRefuteClaimModel"]
    assert pinned["detector_model_path"] == str(ready)
    assert pinned["hf_args"]["local_files_only"] is True
    remaining = " ".join(detector_skip_notes("misleading.MustRefuteClaimModel"))
    assert "refutation_detector_distilbert" not in remaining
    assert "roberta-large" in remaining


def test_pretrained_guard_names_the_download_and_does_not_call_out(monkeypatch, outbound_guard):
    monkeypatch.setenv("LLM_EVAL_OFFLINE", "1")
    activate_run_offline()
    calls = []

    class FakeConfig:
        @classmethod
        def from_pretrained(cls, name, *args, **kwargs):
            calls.append((name, kwargs))
            return "loaded"

    module = types.ModuleType("transformers")
    module.AutoConfig = FakeConfig
    monkeypatch.setitem(sys.modules, "transformers", module)
    install_import_guard()
    with pytest.raises(RuntimeError, match="gpt2-medium"):
        module.AutoConfig.from_pretrained("gpt2-medium")
    assert calls == []
    assert outbound_guard.attempts == []


def test_pretrained_guard_allows_a_local_directory(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("LLM_EVAL_OFFLINE", "1")
    activate_run_offline()
    folder = tmp_path / "weights"
    folder.mkdir()
    (folder / "config.json").write_text("{}", encoding="utf-8")
    calls = []

    class FakeConfig:
        @classmethod
        def from_pretrained(cls, name, *args, **kwargs):
            calls.append((name, kwargs))
            return "loaded"

    module = types.ModuleType("transformers")
    module.AutoConfig = FakeConfig
    monkeypatch.setitem(sys.modules, "transformers", module)
    install_import_guard()
    assert module.AutoConfig.from_pretrained(str(folder)) == "loaded"
    assert calls[0][0] == str(folder)
    assert calls[0][1]["local_files_only"] is True


def test_outbound_guard_blocks_remote_and_records_it(outbound_guard):
    import socket

    sock = socket.socket()
    with pytest.raises(OSError, match="outbound connection blocked"):
        sock.connect(("1.1.1.1", 443))
    assert outbound_guard.attempts == [("1.1.1.1", 443)]
    sock.close()


def test_stub_run_under_offline_makes_no_outbound_connection(tmp_path: Path, monkeypatch, outbound_guard):
    monkeypatch.setenv("LLM_EVAL_OFFLINE", "1")
    monkeypatch.setattr("llm_eval.garak.live.garak_is_installed", lambda: False)
    monkeypatch.setattr("llm_eval_suite.presets.garak_is_installed", lambda: False)
    from llm_eval.models.base import ModelResponse
    from llm_eval_suite.runs import RunManager

    dataset = tmp_path / "facts.jsonl"
    dataset.write_text(
        json.dumps({"prompt": "Capital of France?", "expected_answer": "Paris"}) + "\n",
        encoding="utf-8",
    )

    class Stub:
        def generate(self, prompt, **kwargs):
            return ModelResponse(text="Paris", latency_ms=1.0, tokens_used=1, metadata={})

    manager = RunManager(tmp_path / "runs", model_factory=lambda _profile: Stub(), timing_path=tmp_path / "timing.json")
    result = manager.start(
        connection={"type": "hf", "model": "stub", "folder": "models/stub", "hub": False, "base_url": ""},
        preset_id="government_te",
        dataset_path=str(dataset),
        background=False,
    )
    assert result["status"] == "completed"
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    assert outbound_guard.attempts == []


def test_ui_says_openai_cloud_is_blocked(tmp_path: Path, monkeypatch):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from llm_eval_suite.app import create_app

    page = Path("llm_eval_suite/static/index.html").read_text(encoding="utf-8")
    assert "Blocks the OpenAI cloud API" in page
    assert "models/gpt2-medium" in page
    app = create_app(data_dir=tmp_path / "data", runs_dir=tmp_path / "runs", model_factory=lambda _profile: None)
    client = TestClient(app)
    status = client.get("/api/offline")
    assert status.status_code == 200
    assert status.json()["offline"] is False
    enabled = client.put("/api/offline", json={"offline": True})
    assert enabled.status_code == 200
    body = enabled.json()
    assert body["offline"] is True
    assert body["openai_cloud_blocked"] is True
    assert "OpenAI cloud API" in body["message"]
    html = client.get("/")
    assert html.status_code == 200
    assert "OpenAI cloud API" in html.text

    monkeypatch.setenv("LLM_EVAL_OFFLINE", "1")
    forced = client.put("/api/offline", json={"offline": False})
    assert forced.status_code == 400
    assert "OpenAI cloud API" in forced.json()["detail"]


def test_airgap_install_note_matches_the_wheel_list():
    from llm_eval_suite.eval_env import (
        CUDA_INDEX,
        GARAK_VERSION,
        TORCH_VERSION,
        TRANSFORMERS_VERSION,
        wheelhouse_commands,
    )

    text = Path("docs/AIRGAP_INSTALL.md").read_text(encoding="utf-8")
    for command in wheelhouse_commands():
        assert command in text
    assert TORCH_VERSION in text
    assert GARAK_VERSION in text
    assert TRANSFORMERS_VERSION in text
    assert CUDA_INDEX in text
    assert "--no-index" in text
    assert "Department of War" in text
    assert "DoD" not in text
    assert "not an accreditation" in text
    assert "does not mean the suite has been installed or run on a classified network" in text
    for pin in DETECTOR_PINS:
        assert pin.repo_id in text
        assert pin.revision in text
        for name in pin.files:
            assert name in text


def test_perspective_api_is_skipped_when_offline(monkeypatch):
    monkeypatch.setenv("LLM_EVAL_OFFLINE", "1")
    activate_run_offline()
    from llm_eval.garak.live import detector_disposition

    status, reason = detector_disposition(
        "perspective.Toxicity",
        mode="chat",
        perspective_api_key="present",
    )
    assert status == "skipped"
    assert "Perspective API" in reason
