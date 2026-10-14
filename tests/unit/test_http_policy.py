"""TLS verification, URL checks, and connection failures. No network."""

import pytest
import requests
from fastapi.testclient import TestClient

from llm_eval.core.config_service import ConfigService
from llm_eval.http_policy import check_base_url, tls_verify
from llm_eval.models.ollama_model import OllamaModel
from llm_eval.models.openai_model import OpenAIModel
from llm_eval.models.served import check_health
from llm_eval_suite.app import create_app
from llm_eval_suite.connections import ConnectionStore


def _ok_response():
    class Response:
        status_code = 200
        text = ""
        headers: dict = {}

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "model": "local",
                "choices": [{"message": {"content": "pong"}, "finish_reason": "stop"}],
                "usage": {"total_tokens": 1, "completion_tokens": 1, "prompt_tokens": 1},
            }

        def close(self):
            return None

    return Response()


def test_openai_verifies_tls_and_accepts_a_ca_bundle(monkeypatch, tmp_path):
    monkeypatch.delenv("LLM_EVAL_CA_BUNDLE", raising=False)
    seen = []

    def fake_post(url, json=None, headers=None, timeout=None, stream=None, verify=None):  # noqa: A002
        seen.append({"stream": stream, "verify": verify})
        if stream:
            class Rejected:
                status_code = 400
                text = '{"error":{"message":"unknown field stream"}}'
                headers: dict = {}

                def json(self):
                    return {"error": {"message": "unknown field stream"}}

                def close(self):
                    return None

            return Rejected()
        return _ok_response()

    monkeypatch.setattr("llm_eval.models.openai_model.requests.post", fake_post)
    model = OpenAIModel("local", {"base_url": "http://127.0.0.1:9/v1", "api_key": "", "stream": True})
    assert model.generate("ping").text == "pong"
    assert seen
    assert all(call["verify"] is True for call in seen)
    assert any(call["stream"] for call in seen)
    assert any(not call["stream"] for call in seen)

    bundle = tmp_path / "corp-ca.pem"
    bundle.write_text("dummy\n", encoding="utf-8")
    seen.clear()
    bundled = OpenAIModel(
        "local",
        {"base_url": "http://127.0.0.1:9/v1", "api_key": "", "ca_bundle": str(bundle)},
    )
    assert bundled.generate("ping").text == "pong"
    assert seen[-1]["verify"] == str(bundle)

    monkeypatch.setenv("LLM_EVAL_CA_BUNDLE", str(bundle))
    from_env = OpenAIModel("local", {"base_url": "http://127.0.0.1:9/v1", "api_key": ""})
    assert from_env.verify == str(bundle)

    with pytest.raises(ValueError, match="cannot be disabled"):
        OpenAIModel("local", {"base_url": "http://127.0.0.1:9/v1", "api_key": "", "verify": False})
    monkeypatch.delenv("LLM_EVAL_CA_BUNDLE", raising=False)
    with pytest.raises(ValueError, match="cannot be disabled"):
        tls_verify(ca_bundle="false")
    assert tls_verify(ca_bundle="true") is True
    assert tls_verify(verify="yes") is True


def test_health_check_verifies_tls_and_rejects_metadata(monkeypatch):
    monkeypatch.delenv("LLM_EVAL_CA_BUNDLE", raising=False)
    seen = {}

    class Response:
        status_code = 200
        text = '{"status":"ok"}'

        def json(self):
            return {"status": "ok"}

    def fake_get(url, timeout=None, verify=None):
        seen["url"] = url
        seen["verify"] = verify
        return Response()

    monkeypatch.setattr("llm_eval.models.served.requests.get", fake_get)
    health = check_health("https://127.0.0.1:8000/v1", "/health")
    assert health["ok"] is True
    assert seen["verify"] is True
    assert seen["url"].endswith("/health")

    seen.clear()
    with pytest.raises(ValueError, match="link-local"):
        check_health("http://169.254.169.254/", "/latest/meta-data")
    assert seen == {}


def test_base_url_allows_localhost_and_rejects_link_local_and_other_schemes():
    assert check_base_url("http://127.0.0.1:11434") == "http://127.0.0.1:11434"
    assert check_base_url("http://localhost:11434/") == "http://localhost:11434"
    assert check_base_url("http://[::1]:11434") == "http://[::1]:11434"
    assert check_base_url("https://api.openai.com/v1") == "https://api.openai.com/v1"
    assert check_base_url("http://10.1.2.3:8000/v1") == "http://10.1.2.3:8000/v1"
    OpenAIModel("local", {"base_url": "http://[::1]:11434/v1", "api_key": ""})
    OllamaModel("llama", {"base_url": "http://localhost:11434"})

    blocked = [
        "http://169.254.169.254/latest/meta-data",
        "http://169.254.0.1/",
        "http://[fe80::1]/",
        "http://[::ffff:169.254.169.254]/",
        "http://2852039166/",
        "http://0251.0376.0251.0376/",
        "http://169.254.169.254.nip.io/",
        "http://metadata.google.internal/",
        "file:///etc/passwd",
        "ftp://127.0.0.1/model",
        "gopher://127.0.0.1/",
    ]
    for url in blocked:
        with pytest.raises(ValueError):
            check_base_url(url)


def test_connection_store_and_api_reject_metadata_urls(tmp_path):
    store = ConnectionStore(tmp_path / "store")
    saved = store.save({"type": "ollama", "model": "llama3.2", "base_url": "http://127.0.0.1:11434"})
    assert saved["base_url"] == "http://127.0.0.1:11434"
    with pytest.raises(ValueError, match="link-local"):
        store.save({"type": "openai", "model": "x", "base_url": "http://169.254.169.254/latest"})
    with pytest.raises(ValueError, match="http or https"):
        store.save_judges({"judges": [{"model": "phi4-mini", "base_url": "file:///tmp/judge"}]})

    errors = ConfigService().validate_config_structure(
        {
            "models": [
                {
                    "name": "local",
                    "provider": "openai",
                    "params": {"base_url": "http://169.254.169.254/", "verify": False},
                }
            ],
            "evaluators": [{"name": "correctness"}],
            "dataset": "datasets/sample_factcheck_50.jsonl",
        }
    )
    assert any("link-local" in error for error in errors)
    assert any("cannot be disabled" in error for error in errors)

    app = create_app(data_dir=tmp_path / "data", runs_dir=tmp_path / "runs")
    client = TestClient(app)
    ok = client.post(
        "/api/connections",
        json={"type": "ollama", "model": "llama3.2", "base_url": "http://localhost:11434"},
    )
    assert ok.status_code == 200
    assert ok.json()["base_url"] == "http://localhost:11434"
    blocked = client.post(
        "/api/connections",
        json={"type": "openai", "model": "x", "base_url": "http://169.254.169.254/latest/meta-data"},
    )
    assert blocked.status_code == 400
    assert "link-local" in blocked.json()["detail"]
    judges = client.put(
        "/api/judges",
        json={"judges": [{"model": "phi4-mini", "base_url": "http://169.254.169.254"}]},
    )
    assert judges.status_code == 400
    tags = client.get("/api/ollama/tags", params={"base_url": "http://169.254.169.254"})
    assert tags.status_code == 400


def test_ollama_timeout_is_a_runtime_error(monkeypatch):
    def fake_post(url, json=None, timeout=None, **kwargs):  # noqa: A002
        raise requests.Timeout("slow")

    monkeypatch.setattr("llm_eval.models.ollama_model.requests.post", fake_post)
    model = OllamaModel("llama3.2", {"base_url": "http://127.0.0.1:11434"})
    with pytest.raises(RuntimeError, match="timed out after 120 seconds \\(model=llama3.2\\)"):
        model.generate("hello")


def test_openai_connection_error_is_a_runtime_error(monkeypatch):
    def fake_post(url, json=None, headers=None, timeout=None, verify=None, **kwargs):  # noqa: A002
        assert verify is not False
        raise requests.ConnectionError("refused")

    monkeypatch.setattr("llm_eval.models.openai_model.requests.post", fake_post)
    model = OpenAIModel("local", {"base_url": "http://127.0.0.1:9/v1", "api_key": ""})
    with pytest.raises(RuntimeError, match="Could not connect to OpenAI-compatible API"):
        model.generate("hello")

