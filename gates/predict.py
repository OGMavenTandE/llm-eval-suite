"""Run a model over gate items and save its predictions.

Any model the suite already connects to works: Ollama, OpenAI-compatible,
vLLM, llama.cpp server, or a local Hugging Face folder (for example a LoRA
adapter merged by ``gates.train_lora --merge``). Raw replies are kept next to
the parsed label so a reviewer can see why an item was scored the way it was.
"""

from __future__ import annotations

import time
from pathlib import Path

from gates.data import parse_label, write_jsonl
from gates.spec import GateSpec


def build_model(provider: str, name: str, params: dict):
    if provider == "hf":
        from llm_eval.models.hf_folder import HuggingFaceFolderModel

        return HuggingFaceFolderModel(name, params)
    from llm_eval.models import MODEL_REGISTRY

    if provider not in MODEL_REGISTRY:
        raise ValueError(f"unknown provider {provider!r}; use one of {sorted(MODEL_REGISTRY) + ['hf']}")
    return MODEL_REGISTRY[provider](name, params)


def predict(spec: GateSpec, items: list[dict], model, *, out_path: str | Path, progress=print) -> list[dict]:
    rows = []
    started = time.monotonic()
    for i, item in enumerate(items, 1):
        reply, error, latency, tokens = None, None, None, None
        try:
            response = model.generate(spec.prompt_for(item["text"]))
            reply, latency, tokens = response.text, response.latency_ms, response.tokens_used
        except Exception as exc:  # one failed call is recorded, not fatal
            error = f"{type(exc).__name__}: {exc}"
        rows.append(
            {
                "id": item["id"],
                "reply": reply,
                "prediction": parse_label(reply, spec.labels),
                "latency_ms": latency,
                "tokens_used": tokens,
                "error": error,
                "model": getattr(model, "name", ""),
            }
        )
        if progress and (i % 25 == 0 or i == len(items)):
            progress(f"[predict] {i}/{len(items)} {time.monotonic() - started:.0f}s")
    write_jsonl(out_path, rows)
    return rows
