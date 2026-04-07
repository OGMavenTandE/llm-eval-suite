import time

import requests

from llm_eval.models.base import BaseModel, ModelResponse


class OllamaModel(BaseModel):
    def __init__(self, name: str, params: dict):
        super().__init__(name, params)
        self.base_url = params.get("base_url", "http://localhost:11434").rstrip("/")

    def generate(self, prompt: str, **kwargs) -> ModelResponse:
        payload = {
            "model": self.name,
            "prompt": prompt,
            "stream": False,
            **{k: v for k, v in self.params.items() if k != "base_url"},
            **kwargs,
        }

        url = f"{self.base_url}/api/generate"
        start = time.perf_counter()

        try:
            resp = requests.post(url, json=payload, timeout=120)
            latency_ms = (time.perf_counter() - start) * 1000
            resp.raise_for_status()
        except requests.exceptions.ConnectionError:
            raise RuntimeError(
                f"Could not connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running (`ollama serve`)."
            )
        except requests.exceptions.Timeout:
            raise RuntimeError(
                f"Request to Ollama timed out after 120 seconds (model={self.name})."
            )
        except requests.exceptions.HTTPError as e:
            raise RuntimeError(
                f"Ollama API error {resp.status_code}: {resp.text}"
            ) from e

        data = resp.json()
        text = data.get("response", "")
        tokens_used = data.get("eval_count")

        metadata = {
            "model": data.get("model"),
            "done": data.get("done"),
            "total_duration_ns": data.get("total_duration"),
            "load_duration_ns": data.get("load_duration"),
            "prompt_eval_count": data.get("prompt_eval_count"),
        }

        return ModelResponse(
            text=text,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            metadata=metadata,
        )
