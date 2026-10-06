import time

import requests

from llm_eval.models.base import BaseModel, ModelResponse
from llm_eval.models.context import strip_think_blocks


def judge_num_ctx(prompt: str, max_tokens: int) -> int:
    """Context window for one judge call.

    Ollama's default context is smaller than a chairman prompt, so the reply
    can come back empty. Size the window to the prompt, with a floor above
    that default and a cap so a huge prompt cannot request an unbounded context.
    """
    estimate = (len(prompt) // 4) + int(max_tokens) + 256
    return min(32768, max(4096, estimate))


def ollama_chat(
    base_url: str,
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    think: bool = False,
    timeout: int = 120,
) -> dict:
    """POST native ``/api/chat``. ``think: false`` is a field on this route.

    The OpenAI-compatible ``/v1/chat/completions`` route often ignores ``think``
    and returns the trace in ``message.thinking`` with an empty ``content``.
    This function reads ``content`` only. The thinking field is returned so
    callers can see it was present, and is not copied into ``text``.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": think,
        "options": {
            "num_predict": int(max_tokens),
            "num_ctx": judge_num_ctx(prompt, max_tokens),
        },
    }
    url = f"{base_url.rstrip('/')}/api/chat"
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Could not connect to Ollama at {base_url}. "
            "Make sure Ollama is running (`ollama serve`)."
        ) from None
    except requests.exceptions.Timeout:
        raise RuntimeError(
            f"Request to Ollama timed out after {timeout} seconds (model={model})."
        ) from None
    except requests.exceptions.HTTPError as exc:
        raise RuntimeError(f"Ollama API error {resp.status_code}: {resp.text}") from exc

    data = resp.json()
    message = data.get("message") or {}
    content = message.get("content") or ""
    return {
        "text": strip_think_blocks(content),
        "done_reason": data.get("done_reason"),
        "thinking": message.get("thinking") or message.get("reasoning") or "",
    }


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
        if "think" not in payload:
            payload["think"] = False

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
        text = strip_think_blocks(data.get("response", ""))
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
