import os
import time
from urllib.parse import urlparse

import requests

from llm_eval.models.base import BaseModel, ModelResponse
from llm_eval.models.context import truncate_to_token_budget

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
            budget = max(1, int(self.max_context) - max_new)
            prompt = truncate_to_token_budget(prompt, budget)
            max_new = min(max_new, max(1, int(self.max_context) - 1))
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

        start = time.perf_counter()
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=self.timeout)
            latency_ms = (time.perf_counter() - start) * 1000
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Could not connect to OpenAI-compatible API at {self.base_url}."
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Request timed out after {self.timeout} seconds (model={self.name})."
            )
        except requests.exceptions.HTTPError as exc:
            try:
                error_detail = resp.json().get("error", {}).get("message", resp.text)
            except Exception:
                error_detail = resp.text
            raise RuntimeError(f"API error {resp.status_code}: {error_detail}") from exc

        data = resp.json()
        choice = data["choices"][0]
        if self.mode == "completions":
            text = choice.get("text") or ""
        else:
            message = choice.get("message") or {}
            text = message.get("content") or choice.get("text") or ""

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
