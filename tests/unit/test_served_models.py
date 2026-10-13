"""vLLM and llama.cpp connectors against a local mock server. No GPU."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread

import pytest

from llm_eval.core.config_service import SUPPORTED_PROVIDERS, ConfigService
from llm_eval.models import MODEL_REGISTRY
from llm_eval.models.served import (
    SERVER_PRESETS,
    LlamaCppModel,
    VllmModel,
    check_health,
    list_models,
)
from llm_eval.runner import _get_model_class


class ScriptedHandler(BaseHTTPRequestHandler):
    mode = "stream"
    seen: list[dict] = []

    def log_message(self, format, *args):  # noqa: A003
        return

    def _json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        path = self.path.split("?", 1)[0].rstrip("/")
        if path == "/health":
            self._json(200, {"status": "ok"})
            return
        if path.endswith("/models"):
            self._json(200, {"data": [{"id": "local-model"}, {"id": "other"}]})
            return
        self._json(404, {"error": {"message": "not found"}})

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        ScriptedHandler.seen.append(body)
        if body.get("stream") and body.get("stream_options") and ScriptedHandler.mode == "reject_options":
            self._json(400, {"error": {"message": "unknown field stream_options"}})
            return
        if body.get("stream") and ScriptedHandler.mode == "reject_stream":
            self._json(400, {"error": {"message": "unknown field stream"}})
            return
        if body.get("stream") and ScriptedHandler.mode == "json_anyway":
            self._json(
                200,
                {
                    "model": "local-model",
                    "choices": [{"message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}],
                    "usage": {"total_tokens": 2},
                },
            )
            return
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            first = {"model": "local-model", "choices": [{"delta": {"content": "pon"}, "finish_reason": None}]}
            second = {
                "model": "local-model",
                "choices": [{"delta": {"content": "g"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            }
            self.wfile.write(("data: " + json.dumps(first) + "\n\n").encode("utf-8"))
            self.wfile.flush()
            self.wfile.write(("data: " + json.dumps(second) + "\n\n").encode("utf-8"))
            self.wfile.write(b"data: [DONE]\n\n")
            return
        self._json(
            200,
            {
                "model": "local-model",
                "choices": [{"message": {"role": "assistant", "content": "pong"}, "finish_reason": "stop"}],
                "usage": {"total_tokens": 2, "completion_tokens": 1, "prompt_tokens": 1},
            },
        )


@pytest.fixture
def server():
    ScriptedHandler.mode = "stream"
    ScriptedHandler.seen = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), ScriptedHandler)
    thread = Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    host, port = httpd.server_address[:2]
    base = f"http://{host}:{port}/v1"
    yield base
    httpd.shutdown()
    httpd.server_close()


def test_presets_pin_ports_chat_template_and_token_cap():
    vllm = SERVER_PRESETS["vllm"]
    llama = SERVER_PRESETS["llama_cpp"]
    assert vllm.default_base_url == "http://127.0.0.1:8000/v1"
    assert llama.default_base_url == "http://127.0.0.1:8080/v1"
    assert vllm.health_path == "/health"
    assert llama.health_path == "/health"
    assert vllm.default_max_tokens == 512
    assert llama.default_max_tokens == 512
    assert vllm.mode == "chat" and llama.mode == "chat"
    assert vllm.use_chat_template is True
    assert llama.use_chat_template is True
    assert vllm.stream is True and llama.stream is True
    assert "16" in vllm.note
    model = VllmModel("local-model", {})
    other = LlamaCppModel("local.gguf", {})
    assert model.base_url == vllm.default_base_url
    assert other.base_url == llama.default_base_url
    assert model.max_new_tokens == 512
    assert model.stream is True
    assert model.mode == "chat"


def test_registry_and_yaml_accept_both_providers():
    assert MODEL_REGISTRY["vllm"] is VllmModel
    assert MODEL_REGISTRY["llama_cpp"] is LlamaCppModel
    assert _get_model_class("vllm") is VllmModel
    assert _get_model_class("llama_cpp") is LlamaCppModel
    assert "vllm" in SUPPORTED_PROVIDERS
    assert "llama_cpp" in SUPPORTED_PROVIDERS
    errors = ConfigService().validate_config_structure(
        {
            "models": [{"name": "local-model", "provider": "vllm"}],
            "evaluators": [{"name": "latency"}],
            "dataset": "datasets/sample_factcheck_50.jsonl",
        }
    )
    assert errors == []
    with pytest.raises(ValueError, match="Unknown model provider"):
        _get_model_class("missing")


def test_stream_reports_ttft_and_sends_the_token_cap(server: str):
    model = VllmModel("local-model", {"base_url": server})
    result = model.generate("ping")
    assert result.text == "pong"
    assert result.metadata["ttft_ms"] is not None
    assert result.metadata["streaming"] is True
    assert result.metadata["chat_template"] == "server"
    assert result.metadata["provider"] == "vllm"
    assert result.tokens_used == 2
    body = ScriptedHandler.seen[0]
    assert body["max_tokens"] == 512
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert body["messages"][0]["content"] == "ping"
    llama = LlamaCppModel("local-model", {"base_url": server, "max_new_tokens": 128})
    ScriptedHandler.seen = []
    llama_result = llama.generate("ping")
    assert llama_result.text == "pong"
    assert llama_result.metadata["chat_template"] == "server"
    assert ScriptedHandler.seen[0]["max_tokens"] == 128


def test_completions_mode_skips_the_chat_template(server: str):
    model = LlamaCppModel("local-model", {"base_url": server, "mode": "completions"})
    result = model.generate("ping")
    assert result.metadata["chat_template"] == "off"
    assert "prompt" in ScriptedHandler.seen[0]
    assert "messages" not in ScriptedHandler.seen[0]


def test_stream_options_rejection_retries_without_them(server: str):
    ScriptedHandler.mode = "reject_options"
    model = VllmModel("local-model", {"base_url": server})
    result = model.generate("ping")
    assert result.text == "pong"
    assert result.metadata["streaming"] is True
    assert len(ScriptedHandler.seen) == 2
    assert "stream_options" in ScriptedHandler.seen[0]
    assert "stream_options" not in ScriptedHandler.seen[1]
    assert ScriptedHandler.seen[1]["stream"] is True


def test_stream_rejection_falls_back_to_json(server: str):
    ScriptedHandler.mode = "reject_stream"
    model = VllmModel("local-model", {"base_url": server})
    result = model.generate("ping")
    assert result.text == "pong"
    assert result.metadata.get("streaming") is not True
    assert len(ScriptedHandler.seen) == 2
    assert ScriptedHandler.seen[0]["stream"] is True
    assert "stream" not in ScriptedHandler.seen[1]


def test_json_body_on_a_stream_request_is_still_read(server: str):
    ScriptedHandler.mode = "json_anyway"
    model = VllmModel("local-model", {"base_url": server})
    result = model.generate("ping")
    assert result.text == "pong"
    assert result.metadata["streaming"] is False


def test_health_and_model_list(server: str):
    health = check_health(server, "/health")
    assert health["ok"] is True
    assert health["url"].endswith("/health")
    assert "/v1/health" not in health["url"]
    listed = list_models(server)
    assert listed["ok"] is True
    assert listed["models"] == ["local-model", "other"]
    closed = check_health("http://127.0.0.1:9/v1", "/health", timeout=0.5)
    assert closed["ok"] is False
    assert closed["error"]


def test_connection_save_and_api_round_trip(server: str, tmp_path: Path):
    from fastapi.testclient import TestClient

    from llm_eval_suite.app import create_app
    from llm_eval_suite.connections import ConnectionStore, build_model

    store = ConnectionStore(tmp_path)
    saved = store.save({"type": "vllm", "model": "local-model", "base_url": "http://127.0.0.1:8000"})
    assert saved["max_new_tokens"] == 512
    assert saved["use_chat_template"] is True
    assert saved["mode"] == "chat"
    blank = store.save({"type": "llama_cpp", "model": "local.gguf", "base_url": ""})
    assert blank["base_url"] == "http://127.0.0.1:8080/v1"
    assert blank["max_new_tokens"] == 512
    built = build_model({"type": "llama_cpp", "model": "local-model", "base_url": server})
    assert built.base_url == server
    assert built.generate("ping").text == "pong"

    app = create_app(data_dir=tmp_path / "data", runs_dir=tmp_path / "runs")
    client = TestClient(app)
    presets = client.get("/api/servers/presets")
    assert presets.status_code == 200
    providers = {row["provider"] for row in presets.json()["presets"]}
    assert providers == {"vllm", "llama_cpp"}
    health = client.get("/api/servers/health", params={"server": "vllm", "base_url": server})
    assert health.status_code == 200
    assert health.json()["ok"] is True
    models = client.get("/api/servers/models", params={"server": "llama_cpp", "base_url": server})
    assert models.status_code == 200
    assert models.json()["models"] == ["local-model", "other"]
    tested = client.post(
        "/api/connections/test",
        json={"profile": {"type": "vllm", "model": "local-model", "base_url": server, "mode": "chat"}},
    )
    assert tested.status_code == 200
    body = tested.json()
    assert body["ok"] is True
    assert body["reply"] == "pong"
    assert body["ttft_ms"] is not None
    assert body["health"]["ok"] is True
    page = client.get("/")
    assert page.status_code == 200
    assert 'value="vllm"' in page.text
    assert 'value="llama_cpp"' in page.text


def test_unhealthy_server_skips_generation(tmp_path: Path):
    from fastapi.testclient import TestClient

    from llm_eval_suite.app import create_app

    app = create_app(data_dir=tmp_path / "data", runs_dir=tmp_path / "runs")
    client = TestClient(app)
    tested = client.post(
        "/api/connections/test",
        json={"profile": {"type": "llama_cpp", "model": "local-model", "base_url": "http://127.0.0.1:9/v1"}},
    )
    assert tested.status_code == 200
    body = tested.json()
    assert body["ok"] is False
    assert body["reply"] == ""
    assert body["health"]["ok"] is False


def test_offline_blocks_a_remote_vllm_url(monkeypatch: pytest.MonkeyPatch):
    from llm_eval.offline import activate_run_offline

    monkeypatch.setenv("LLM_EVAL_OFFLINE", "1")
    activate_run_offline()
    model = VllmModel("local-model", {"base_url": "https://example.com/v1"})
    with pytest.raises(RuntimeError, match="Offline mode"):
        model.generate("ping")


def test_docs_send_tensorrt_and_nim_through_openai():
    text = Path("docs/OPENAI_SERVERS.md").read_text(encoding="utf-8")
    assert "TensorRT-LLM" in text
    assert "NIM" in text
    assert "openai" in text
    assert "/v1/health/ready" in text
    assert "http://127.0.0.1:8000/v1" in text
    assert "http://127.0.0.1:8080/v1" in text
    page = Path("llm_eval_suite/static/app.js").read_text(encoding="utf-8")
    assert "http://127.0.0.1:8000/v1" in page
    assert "http://127.0.0.1:8080/v1" in page
    assert "512" in page


def test_discovery_probe_uses_the_preset(monkeypatch: pytest.MonkeyPatch):
    from apps.api.services.discovery import _apply_server_probe

    monkeypatch.setattr(
        "llm_eval.models.served.check_health",
        lambda *args, **kwargs: {"ok": True},
    )
    monkeypatch.setattr(
        "llm_eval.models.served.list_models",
        lambda *args, **kwargs: {"ok": True, "models": ["local-model"]},
    )
    item = {"connection_status": "unknown", "available": False, "warning_message": None}
    _apply_server_probe(item, "llama_cpp", "local-model")
    assert item["connection_status"] == "reachable"
    assert item["available"] is True
