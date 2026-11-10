"""Streaming TTFT for Ollama, plus the SSE parser used by older tests.

Ollama honors ``edge_stream=True`` here. OpenAI, vLLM, and llama.cpp stream
through ``OpenAIModel._generate_stream`` and report ``ttft_ms`` themselves.
"""

from __future__ import annotations

import json
import time

import requests

from llm_eval.http_policy import check_base_url, refuse_redirect
from llm_eval.models.base import ModelResponse
from llm_eval.models.context import strip_think_blocks
from llm_eval.offline import refuse_remote_http


def consume_ollama_lines(lines, start: float, clock=time.perf_counter) -> tuple[float | None, str, dict, float]:
    """First non-empty ``response`` chunk sets TTFT. The last JSON object is the summary."""
    ttft = None
    parts: list[str] = []
    last: dict = {}
    for raw in lines:
        if not raw:
            continue
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        data = json.loads(raw)
        piece = data.get("response") or ""
        if piece and ttft is None:
            ttft = (clock() - start) * 1000
        if piece:
            parts.append(piece)
        last = data
    latency_ms = (clock() - start) * 1000
    return ttft, "".join(parts), last, latency_ms


def consume_openai_sse(lines, start: float, mode: str, clock=time.perf_counter):
    """First content delta sets TTFT. ``[DONE]`` ends the stream."""
    ttft = None
    parts: list[str] = []
    usage: dict = {}
    finish = None
    model = None
    for raw in lines:
        if not raw:
            continue
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8", errors="replace")
        line = raw.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        data = json.loads(payload)
        model = data.get("model") or model
        if data.get("usage"):
            usage = data["usage"]
        choices = data.get("choices") or [{}]
        choice = choices[0] if choices else {}
        finish = choice.get("finish_reason") or finish
        if mode == "completions":
            piece = choice.get("text") or ""
        else:
            delta = choice.get("delta") or {}
            piece = delta.get("content") or ""
            if not piece and choice.get("message"):
                piece = (choice.get("message") or {}).get("content") or ""
        if piece and ttft is None:
            ttft = (clock() - start) * 1000
        if piece:
            parts.append(piece)
    latency_ms = (clock() - start) * 1000
    return ttft, "".join(parts), usage, finish, model, latency_ms


def _response(text: str, latency_ms: float, tokens, metadata: dict) -> ModelResponse:
    return ModelResponse(
        text=strip_think_blocks(text),
        latency_ms=latency_ms,
        tokens_used=tokens,
        metadata=metadata,
    )


def ollama_stream_generate(base_url: str, payload: dict, timeout: int = 120) -> ModelResponse:
    """Stream ``/api/generate``. On a bad stream body, one non-streaming call is used."""
    base_url = check_base_url(base_url)
    refuse_remote_http(base_url, what="remote Ollama")
    url = f"{base_url}/api/generate"
    body = {key: value for key, value in payload.items() if key != "edge_stream"}
    body["stream"] = True
    model = body.get("model") or ""
    start = time.perf_counter()
    try:
        resp = requests.post(url, json=body, timeout=timeout, stream=True, allow_redirects=False)
        refuse_redirect(resp)
        resp.raise_for_status()
        ttft, text, last, latency_ms = consume_ollama_lines(resp.iter_lines(), start)
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
        detail = getattr(getattr(exc, "response", None), "text", "") or str(exc)
        code = getattr(getattr(exc, "response", None), "status_code", "")
        raise RuntimeError(f"Ollama API error {code}: {detail}") from exc
    except Exception as exc:
        return _ollama_blocking(url, body, timeout, model, ttft_reason=f"streaming failed: {exc}")

    eval_count = last.get("eval_count")
    eval_ns = last.get("eval_duration")
    reason = "" if ttft is not None else "stream returned no token"
    metadata = {
        "model": last.get("model"),
        "done": last.get("done"),
        "total_duration_ns": last.get("total_duration"),
        "load_duration_ns": last.get("load_duration"),
        "prompt_eval_count": last.get("prompt_eval_count"),
        "eval_duration_ns": eval_ns,
        "eval_seconds": None if not eval_ns else float(eval_ns) / 1e9,
        "ttft_ms": ttft,
        "ttft_reason": reason,
        "output_tokens": eval_count,
        "stream": True,
    }
    return _response(text, latency_ms, eval_count, metadata)


def _ollama_blocking(url: str, body: dict, timeout: int, model: str, ttft_reason: str) -> ModelResponse:
    payload = {key: value for key, value in body.items() if key != "stream"}
    payload["stream"] = False
    start = time.perf_counter()
    try:
        resp = requests.post(url, json=payload, timeout=timeout, allow_redirects=False)
        refuse_redirect(resp)
        latency_ms = (time.perf_counter() - start) * 1000
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Could not connect to Ollama at {url}. Make sure Ollama is running (`ollama serve`)."
        ) from None
    except requests.exceptions.Timeout:
        raise RuntimeError(f"Request to Ollama timed out after {timeout} seconds (model={model}).") from None
    except requests.exceptions.HTTPError as exc:
        detail = getattr(resp, "text", str(exc))
        raise RuntimeError(f"Ollama API error {resp.status_code}: {detail}") from exc
    data = resp.json()
    eval_count = data.get("eval_count")
    metadata = {
        "model": data.get("model"),
        "done": data.get("done"),
        "total_duration_ns": data.get("total_duration"),
        "load_duration_ns": data.get("load_duration"),
        "prompt_eval_count": data.get("prompt_eval_count"),
        "ttft_ms": None,
        "ttft_reason": ttft_reason,
        "output_tokens": eval_count,
        "stream": False,
    }
    return _response(data.get("response") or "", latency_ms, eval_count, metadata)
