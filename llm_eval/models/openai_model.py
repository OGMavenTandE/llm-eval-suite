import json
import os
import time
from urllib.parse import urlparse

import requests

from llm_eval.models.base import BaseModel, ModelResponse
from llm_eval.offline import refuse_remote_http
from llm_eval.models.context import clamp_prompt_and_new_tokens, strip_think_blocks

# Connection settings are not generation parameters.
def _is_local_base_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"}


_CONNECTION_KEYS = {
    "base_url",
    "api_key",
    "mode",
    "max_context",
    "max_new_tokens",
    "timeout",
    "cloud",
    "folder",
    "think",
    "stream",
    "remote_what",
}


class OpenAIModel(BaseModel):
    """OpenAI-compatible client. The API key is optional for local servers."""

    def __init__(self, name: str, params: dict):
        super().__init__(name, params)
        self.base_url = params.get("base_url", "https://api.openai.com/v1").rstrip("/")
        if params.get("api_key"):
            self.api_key = params["api_key"]
        elif "api_key" in params or _is_local_base_url(self.base_url):
            # Explicit blank, or a local server. Do not attach a stray cloud key.
            self.api_key = ""
        else:
            self.api_key = os.environ.get("OPENAI_API_KEY", "")
        self.mode = params.get("mode", "chat")
        if self.mode not in ("chat", "completions"):
            raise ValueError("mode must be 'chat' or 'completions'")
        self.max_context = params.get("max_context")
        self.max_new_tokens = int(params.get("max_new_tokens") or 64)
        self.timeout = int(params.get("timeout") or 120)
        self.think = params.get("think") if "think" in params else None
        self.stream = bool(params.get("stream", False))
        self.remote_what = params.get("remote_what") or "OpenAI cloud API"

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload_params = {
            key: value
            for key, value in self.params.items()
            if key not in _CONNECTION_KEYS
        }
        for key, value in kwargs.items():
            if key not in _CONNECTION_KEYS:
                payload_params[key] = value

        max_new = int(payload_params.pop("max_tokens", self.max_new_tokens) or self.max_new_tokens)
        if self.max_context:
            prompt, max_new = clamp_prompt_and_new_tokens(prompt, max_new, int(self.max_context))
        payload_params["max_tokens"] = max_new

        if self.mode == "completions":
            payload = {"model": self.name, "prompt": prompt, **payload_params}
            url = f"{self.base_url}/completions"
        else:
            payload = {
                "model": self.name,
                "messages": [{"role": "user", "content": prompt}],
                **payload_params,
            }
            url = f"{self.base_url}/chat/completions"
        if self.think is not None:
            payload["think"] = self.think

        refuse_remote_http(self.base_url, what=self.remote_what)
        start = time.perf_counter()
        if self.stream:
            streamed = self._generate_stream(url, payload, headers, start)
            if streamed is not None:
                return streamed
        resp = self._post(url, payload, headers)
        latency_ms = (time.perf_counter() - start) * 1000

        data = resp.json()
        choice = data["choices"][0]
        if self.mode == "completions":
            text = strip_think_blocks(choice.get("text") or "")
        else:
            message = choice.get("message") or {}
            text = strip_think_blocks(message.get("content") or choice.get("text") or "")

        usage = data.get("usage") or {}
        tokens_used = usage.get("total_tokens")
        metadata = {
            "model": data.get("model"),
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "mode": self.mode,
        }
        return ModelResponse(
            text=text,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            metadata=metadata,
        )

    def _generate_stream(self, url: str, payload: dict, headers: dict, start: float, *, allow_options_retry: bool = True):
        """Stream one completion. ``None`` means the server rejected streaming."""
        body = dict(payload)
        body["stream"] = True
        if allow_options_retry:
            body["stream_options"] = {"include_usage": True}
        try:
            resp = requests.post(url, json=body, headers=headers, timeout=self.timeout, stream=True)
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Could not connect to OpenAI-compatible API at {self.base_url}."
            ) from None
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Request timed out after {self.timeout} seconds (model={self.name})."
            ) from None
        if resp.status_code >= 400:
            detail = _error_detail(resp)
            resp.close()
            problem = _stream_problem(detail)
            if resp.status_code == 400 and problem == "options" and allow_options_retry:
                return self._generate_stream(url, payload, headers, start, allow_options_retry=False)
            if resp.status_code == 400 and problem == "stream":
                return None
            raise RuntimeError(f"API error {resp.status_code}: {detail}")
        try:
            content_type = (resp.headers.get("Content-Type") or "").lower()
            if "application/json" in content_type and "event-stream" not in content_type:
                data = resp.json()
                return _response_from_body(self, data, start, streaming=False, ttft_ms=None)
            text, ttft, usage, finish, model_name = _read_sse(resp, start)
            latency_ms = (time.perf_counter() - start) * 1000
            return ModelResponse(
                text=strip_think_blocks(text),
                latency_ms=latency_ms,
                tokens_used=(usage or {}).get("total_tokens"),
                metadata={
                    "model": model_name,
                    "finish_reason": finish,
                    "prompt_tokens": (usage or {}).get("prompt_tokens"),
                    "completion_tokens": (usage or {}).get("completion_tokens"),
                    "mode": self.mode,
                    "streaming": True,
                    "ttft_ms": round(ttft, 2) if ttft is not None else None,
                },
            )
        finally:
            resp.close()

    def _post(self, url: str, payload: dict, headers: dict):
        """POST once. If the server rejects ``think``, retry without it."""
        resp = self._post_once(url, payload, headers)
        if resp is not None:
            return resp
        if "think" not in payload:
            raise RuntimeError(f"Request to {self.base_url} failed.")
        reduced = {key: value for key, value in payload.items() if key != "think"}
        retried = self._post_once(url, reduced, headers)
        if retried is None:
            raise RuntimeError(f"Request to {self.base_url} failed.")
        return retried

    def _post_once(self, url: str, payload: dict, headers: dict):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            resp.raise_for_status()
            return resp
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Could not connect to OpenAI-compatible API at {self.base_url}."
            ) from None
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Request timed out after {self.timeout} seconds (model={self.name})."
            ) from None
        except requests.exceptions.HTTPError as exc:
            try:
                error_detail = resp.json().get("error", {}).get("message", resp.text)
            except Exception:
                error_detail = getattr(resp, "text", str(exc))
            if "think" in payload and _think_rejected(str(error_detail)):
                return None
            raise RuntimeError(f"API error {resp.status_code}: {error_detail}") from exc


def _think_rejected(message: str) -> bool:
    lowered = (message or "").lower()
    if "think" not in lowered:
        return False
    return any(token in lowered for token in ("unknown", "unexpected", "extra", "invalid", "unrecognized", "not permitted"))


def _stream_problem(message: str) -> str:
    """``options`` or ``stream`` when a 400 says that field was rejected."""
    lowered = (message or "").lower()
    tokens = ("unknown", "unexpected", "extra", "invalid", "unrecognized", "not permitted", "not supported")
    if not any(token in lowered for token in tokens):
        return ""
    if "stream_options" in lowered:
        return "options"
    if "stream" in lowered:
        return "stream"
    return ""


def _error_detail(resp) -> str:
    try:
        return str(resp.json().get("error", {}).get("message", resp.text))
    except Exception:
        return str(getattr(resp, "text", ""))


def _read_sse(resp, start: float):
    parts: list[str] = []
    ttft = None
    usage: dict = {}
    finish = None
    model_name = None
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw:
            continue
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            event = json.loads(data)
        except json.JSONDecodeError:
            continue
        model_name = event.get("model") or model_name
        if event.get("usage"):
            usage = event["usage"]
        choices = event.get("choices") or []
        if not choices:
            continue
        choice = choices[0]
        delta = choice.get("delta") or {}
        piece = delta.get("content")
        if piece is None:
            message = choice.get("message") or {}
            piece = message.get("content") or choice.get("text") or ""
        if piece:
            if ttft is None:
                ttft = (time.perf_counter() - start) * 1000
            parts.append(str(piece))
        if choice.get("finish_reason"):
            finish = choice["finish_reason"]
    return "".join(parts), ttft, usage, finish, model_name


def _response_from_body(model: OpenAIModel, data: dict, start: float, *, streaming: bool, ttft_ms):
    choice = data["choices"][0]
    if model.mode == "completions":
        text = strip_think_blocks(choice.get("text") or "")
    else:
        message = choice.get("message") or {}
        text = strip_think_blocks(message.get("content") or choice.get("text") or "")
    usage = data.get("usage") or {}
    latency_ms = (time.perf_counter() - start) * 1000
    return ModelResponse(
        text=text,
        latency_ms=latency_ms,
        tokens_used=usage.get("total_tokens"),
        metadata={
            "model": data.get("model"),
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "mode": model.mode,
            "streaming": streaming,
            "ttft_ms": ttft_ms,
        },
    )
