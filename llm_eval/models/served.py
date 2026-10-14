"""vLLM and llama.cpp server presets.

Both speak the OpenAI chat API. The presets pin the local port, chat mode
so the server applies its chat template, a 512 token cap, and streaming so
a call can report time to first token.

vLLM's own default for ``max_tokens`` is 16 when the client omits it, so
this client always sends a cap. llama.cpp applies the GGUF chat template
on ``/v1/chat/completions``. Completions mode skips that template.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

import requests

from llm_eval.http_policy import check_base_url, tls_verify
from llm_eval.models.base import ModelResponse
from llm_eval.models.openai_model import OpenAIModel
from llm_eval.offline import refuse_remote_http


@dataclass(frozen=True)
class ServerPreset:
    provider: str
    label: str
    default_base_url: str
    health_path: str
    default_max_tokens: int
    mode: str
    use_chat_template: bool
    stream: bool
    note: str

    def as_dict(self) -> dict:
        return {
            "provider": self.provider,
            "label": self.label,
            "default_base_url": self.default_base_url,
            "health_path": self.health_path,
            "default_max_tokens": self.default_max_tokens,
            "mode": self.mode,
            "use_chat_template": self.use_chat_template,
            "stream": self.stream,
            "note": self.note,
        }


SERVER_PRESETS: dict[str, ServerPreset] = {
    "vllm": ServerPreset(
        provider="vllm",
        label="vLLM",
        default_base_url="http://127.0.0.1:8000/v1",
        health_path="/health",
        default_max_tokens=512,
        mode="chat",
        use_chat_template=True,
        stream=True,
        note=(
            "vLLM applies the model chat template on /v1/chat/completions. "
            "max_tokens is sent on every call because vLLM's own default is 16."
        ),
    ),
    "llama_cpp": ServerPreset(
        provider="llama_cpp",
        label="llama.cpp server",
        default_base_url="http://127.0.0.1:8080/v1",
        health_path="/health",
        default_max_tokens=512,
        mode="chat",
        use_chat_template=True,
        stream=True,
        note=(
            "llama.cpp applies the GGUF chat template on /v1/chat/completions. "
            "Start the server with that template, or pass --chat-template."
        ),
    ),
}


def preset_for(provider: str) -> ServerPreset:
    try:
        return SERVER_PRESETS[provider]
    except KeyError as exc:
        known = ", ".join(SERVER_PRESETS)
        raise ValueError(f"Unknown server preset '{provider}'. Known: {known}.") from exc


def server_origin(base_url: str) -> str:
    parsed = urlparse(base_url)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Base URL is not absolute: {base_url}")
    return f"{parsed.scheme}://{parsed.netloc}"


def api_root(base_url: str) -> str:
    root = (base_url or "").rstrip("/")
    if not root.endswith("/v1"):
        root = root + "/v1"
    return root


def check_health(base_url: str, health_path: str = "/health", timeout: float = 5) -> dict:
    """GET the server health URL. A connection failure is ``ok: False``, not a raise."""
    check_base_url(base_url)
    origin = server_origin(base_url)
    path = health_path if health_path.startswith("/") else f"/{health_path}"
    url = origin + path
    refuse_remote_http(url, what="remote endpoint")
    try:
        resp = requests.get(url, timeout=timeout, verify=tls_verify())
    except requests.RequestException as exc:
        return {"ok": False, "url": url, "status_code": None, "body": None, "error": str(exc)}
    body = _body(resp)
    ok = resp.status_code == 200
    if isinstance(body, dict):
        status = str(body.get("status") or "").lower()
        if status and status not in {"ok", "healthy", "ready"}:
            ok = False
    return {
        "ok": ok,
        "url": url,
        "status_code": resp.status_code,
        "body": body,
        "error": None if ok else "Health check failed.",
    }


def list_models(base_url: str, timeout: float = 5) -> dict:
    """GET ``/v1/models`` and return the ids."""
    check_base_url(base_url)
    url = api_root(base_url) + "/models"
    refuse_remote_http(url, what="remote endpoint")
    try:
        resp = requests.get(url, timeout=timeout, verify=tls_verify())
        resp.raise_for_status()
        payload = resp.json()
    except requests.RequestException as exc:
        return {"ok": False, "url": url, "models": [], "error": str(exc)}
    names = []
    for row in payload.get("data") or []:
        model_id = row.get("id") or row.get("name")
        if model_id:
            names.append(str(model_id))
    return {"ok": True, "url": url, "models": names, "error": None}


def _body(resp):
    try:
        return resp.json()
    except ValueError:
        text = (resp.text or "").strip()
        return text[:200] if text else None


class _ServedModel(OpenAIModel):
    preset_key = ""

    def __init__(self, name: str, params: dict):
        preset = SERVER_PRESETS[self.preset_key]
        merged = dict(params)
        if not merged.get("base_url"):
            merged["base_url"] = preset.default_base_url
        if not merged.get("mode"):
            merged["mode"] = preset.mode
        if not merged.get("max_new_tokens"):
            merged["max_new_tokens"] = preset.default_max_tokens
        if "stream" not in merged:
            merged["stream"] = preset.stream
        if "api_key" not in merged:
            merged["api_key"] = ""
        merged.setdefault("remote_what", "remote endpoint")
        super().__init__(name, merged)
        self.preset = preset

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        result = super().generate(prompt, **kwargs)
        result.metadata["provider"] = self.preset.provider
        result.metadata["chat_template"] = "server" if self.mode == "chat" else "off"
        return result


class VllmModel(_ServedModel):
    preset_key = "vllm"


class LlamaCppModel(_ServedModel):
    preset_key = "llama_cpp"


def model_for(provider: str) -> type[_ServedModel]:
    if provider == "vllm":
        return VllmModel
    if provider == "llama_cpp":
        return LlamaCppModel
    raise ValueError(f"Unknown server preset '{provider}'.")
