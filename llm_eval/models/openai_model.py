import os
import time

import requests

from llm_eval.models.base import BaseModel, ModelResponse


class OpenAIModel(BaseModel):
    def __init__(self, name: str, params: dict):
        super().__init__(name, params)
        self.base_url = params.get("base_url", "https://api.openai.com/v1").rstrip("/")
        self.api_key = params.get("api_key") or os.environ.get("OPENAI_API_KEY", "")
        if not self.api_key:
            raise ValueError(
                "No API key provided. Set OPENAI_API_KEY env var or pass api_key in params."
            )

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload_params = {
            k: v for k, v in self.params.items()
            if k not in ("base_url", "api_key")
        }
        payload_params.update(kwargs)

        payload = {
            "model": self.name,
            "messages": [{"role": "user", "content": prompt}],
            **payload_params,
        }

        url = f"{self.base_url}/chat/completions"
        start = time.perf_counter()

        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=120)
            latency_ms = (time.perf_counter() - start) * 1000
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Could not connect to OpenAI-compatible API at {self.base_url}."
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Request timed out after 120 seconds (model={self.name})."
            )
        except requests.exceptions.HTTPError as e:
            try:
                error_detail = resp.json().get("error", {}).get("message", resp.text)
            except Exception:
                error_detail = resp.text
            raise RuntimeError(
                f"API error {resp.status_code}: {error_detail}"
            ) from e

        data = resp.json()
        choice = data["choices"][0]
        text = choice["message"]["content"]

        usage = data.get("usage", {})
        tokens_used = usage.get("total_tokens")

        metadata = {
            "model": data.get("model"),
            "finish_reason": choice.get("finish_reason"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
        }

        return ModelResponse(
            text=text,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            metadata=metadata,
        )
