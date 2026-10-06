"""Named connection profiles stored as JSON under the local data directory."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from urllib.parse import urlparse

from llm_eval.models.base import BaseModel
from llm_eval.models.hf_folder import HuggingFaceFolderModel
from llm_eval.models.nanogpt_convert import describe_model_path
from llm_eval.models.openai_model import OpenAIModel

DEFAULT_JUDGE_TIMEOUT = 90
DEFAULT_JUDGE_MAX_TOKENS = 1200

DEFAULT_JUDGES = [
    {
        "name": "qwen2.5:3b-instruct",
        "type": "ollama",
        "base_url": "http://127.0.0.1:11434",
        "model": "qwen2.5:3b-instruct",
        "mode": "chat",
        "cloud": False,
    },
    {
        "name": "llama3.2:3b",
        "type": "ollama",
        "base_url": "http://127.0.0.1:11434",
        "model": "llama3.2:3b",
        "mode": "chat",
        "cloud": False,
    },
]


def judge_timeout(settings: dict | None = None) -> int:
    """Seconds to wait for one council judge call.

    A saved judge timeout wins, then ``LLM_EVAL_JUDGE_TIMEOUT``, then 90 seconds.
    """
    settings = settings or {}
    chosen = settings.get("timeout")
    if not chosen:
        chosen = os.environ.get("LLM_EVAL_JUDGE_TIMEOUT")
    try:
        seconds = int(chosen) if chosen else DEFAULT_JUDGE_TIMEOUT
    except (TypeError, ValueError):
        seconds = DEFAULT_JUDGE_TIMEOUT
    return max(1, seconds)


def judge_max_tokens(settings: dict | None = None) -> int:
    """Tokens for one council judge reply.

    A saved value wins, then ``LLM_EVAL_JUDGE_MAX_TOKENS``, then 1200.
    Thinking models need the room after ``think`` is turned off.
    """
    settings = settings or {}
    chosen = settings.get("max_tokens")
    if not chosen:
        chosen = os.environ.get("LLM_EVAL_JUDGE_MAX_TOKENS")
    try:
        tokens = int(chosen) if chosen else DEFAULT_JUDGE_MAX_TOKENS
    except (TypeError, ValueError):
        tokens = DEFAULT_JUDGE_MAX_TOKENS
    return max(1, tokens)


def is_local_url(url: str | None) -> bool:
    if not url:
        return True
    host = (urlparse(url).hostname or "").lower()
    return host in {"", "localhost", "127.0.0.1", "::1"}


def redact(profile: dict) -> dict:
    cleaned = dict(profile)
    if cleaned.get("api_key"):
        cleaned["api_key"] = "set"
        cleaned["has_api_key"] = True
    else:
        cleaned["api_key"] = ""
        cleaned["has_api_key"] = False
    return cleaned


class ConnectionStore:
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.data_dir / "connections.json"
        self.judges_path = self.data_dir / "judges.json"

    def list_profiles(self) -> list[dict]:
        return [redact(row) for row in self._read()]

    def get(self, profile_id: str) -> dict | None:
        for row in self._read():
            if row.get("id") == profile_id:
                return row
        return None

    def save(self, payload: dict) -> dict:
        rows = self._read()
        profile_id = payload.get("id") or str(uuid.uuid4())
        existing = next((row for row in rows if row.get("id") == profile_id), None)
        api_key = payload.get("api_key")
        if api_key in (None, "", "set") and existing:
            api_key = existing.get("api_key") or ""
        base_url = (payload.get("base_url") or "").rstrip("/")
        profile = {
            "id": profile_id,
            "name": (payload.get("name") or payload.get("model") or "connection").strip(),
            "type": payload.get("type") or "openai",
            "base_url": base_url,
            "model": (payload.get("model") or "").strip(),
            "api_key": api_key or "",
            "mode": payload.get("mode") or "chat",
            "max_context": payload.get("max_context"),
            "folder": payload.get("folder") or "",
            "cloud": bool(payload.get("cloud")) or not is_local_url(base_url),
            "hub": bool(payload.get("hub")),
            "max_new_tokens": payload.get("max_new_tokens") or 64,
        }
        if profile["type"] not in {"openai", "ollama", "hf", "nanogpt"}:
            raise ValueError("type must be openai, ollama, hf, or nanogpt")
        if profile["type"] in {"openai", "ollama"} and not profile["model"]:
            raise ValueError("A model name is required.")
        if profile["hub"] and not profile["folder"]:
            profile["folder"] = profile["model"]
        if profile["type"] == "hf" and not profile["folder"]:
            raise ValueError("A Hugging Face folder path is required.")
        replaced = False
        for index, row in enumerate(rows):
            if row.get("id") == profile_id:
                rows[index] = profile
                replaced = True
                break
        if not replaced:
            rows.append(profile)
        self._write(rows)
        return redact(profile)

    def delete(self, profile_id: str) -> bool:
        rows = self._read()
        kept = [row for row in rows if row.get("id") != profile_id]
        if len(kept) == len(rows):
            return False
        self._write(kept)
        return True

    def judges(self) -> dict:
        if not self.judges_path.is_file():
            payload = {
                "chairman": DEFAULT_JUDGES[0]["model"],
                "judges": DEFAULT_JUDGES,
                "timeout": DEFAULT_JUDGE_TIMEOUT,
                "max_tokens": DEFAULT_JUDGE_MAX_TOKENS,
            }
            self.judges_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            return payload
        stored = json.loads(self.judges_path.read_text(encoding="utf-8"))
        stored.setdefault("timeout", DEFAULT_JUDGE_TIMEOUT)
        stored.setdefault("max_tokens", DEFAULT_JUDGE_MAX_TOKENS)
        return stored

    def save_judges(self, payload: dict) -> dict:
        judges = []
        for row in payload.get("judges") or []:
            base_url = (row.get("base_url") or "http://127.0.0.1:11434").rstrip("/")
            judges.append(
                {
                    "name": row.get("name") or row.get("model") or "judge",
                    "type": row.get("type") or "ollama",
                    "base_url": base_url,
                    "model": (row.get("model") or "").strip(),
                    "mode": row.get("mode") or "chat",
                    "api_key": row.get("api_key") or "",
                    "cloud": bool(row.get("cloud")) or not is_local_url(base_url),
                }
            )
        chairman = payload.get("chairman") or (judges[0]["model"] if judges else "")
        stored = {
            "chairman": chairman,
            "judges": judges,
            "timeout": judge_timeout(payload),
            "max_tokens": judge_max_tokens(payload),
        }
        self.judges_path.write_text(json.dumps(stored, indent=2) + "\n", encoding="utf-8")
        return stored

    def _read(self) -> list[dict]:
        if not self.path.is_file():
            return []
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return list(data.get("connections") or [])
        return list(data)

    def _write(self, rows: list[dict]) -> None:
        self.path.write_text(
            json.dumps({"connections": rows}, indent=2) + "\n",
            encoding="utf-8",
        )


def openai_base_url(profile: dict) -> str:
    base = (profile.get("base_url") or "").rstrip("/")
    if profile.get("type") == "ollama":
        if not base:
            base = "http://127.0.0.1:11434"
        if not base.endswith("/v1"):
            return base + "/v1"
    if not base:
        return "https://api.openai.com/v1"
    return base


def build_model(profile: dict) -> BaseModel:
    """One adapter interface. Ollama is reached through its ``/v1`` server."""
    kind = profile.get("type") or "openai"
    if kind == "hf":
        return HuggingFaceFolderModel(
            profile.get("model") or Path(profile.get("folder") or "hf").name,
            {
                "folder": profile.get("folder"),
                "max_context": profile.get("max_context") or 1024,
                "mode": profile.get("mode") or "auto",
                "max_new_tokens": profile.get("max_new_tokens") or 64,
                "hub": bool(profile.get("hub")),
            },
        )
    if kind == "nanogpt":
        raise RuntimeError(
            "Convert the nanoGPT checkpoint to a Hugging Face folder, then connect that folder."
        )
    params = {
        "base_url": openai_base_url(profile),
        "api_key": profile.get("api_key") or "",
        "mode": profile.get("mode") or "chat",
        "timeout": profile.get("timeout") or 120,
    }
    if kind == "ollama" or profile.get("think") is not None:
        params["think"] = False if profile.get("think") is None else profile.get("think")
    if profile.get("max_context"):
        params["max_context"] = int(profile["max_context"])
        params["max_new_tokens"] = int(profile.get("max_new_tokens") or 64)
    return OpenAIModel(profile.get("model") or "model", params)


def test_connection(profile: dict, model: BaseModel | None = None, prompt: str | None = None) -> dict:
    prompt = prompt or "Reply with the single word pong."
    try:
        target = model or build_model(profile)
        result = target.generate(prompt, max_tokens=32)
    except Exception as exc:  # noqa: BLE001 - returned to the connect screen
        return {
            "ok": False,
            "reply": "",
            "latency_ms": None,
            "tokens_per_second": None,
            "tokens_used": None,
            "error": str(exc),
        }
    seconds = (result.latency_ms or 0) / 1000
    tokens = result.tokens_used
    rate = None
    if tokens and seconds > 0:
        rate = round(tokens / seconds, 2)
    return {
        "ok": True,
        "reply": result.text,
        "latency_ms": round(result.latency_ms, 2),
        "tokens_per_second": rate,
        "tokens_used": tokens,
        "error": None,
    }


def same_model(judge: dict, under_test: dict) -> bool:
    """The model under test is never a judge, matched on model name."""
    left = (judge.get("model") or "").strip()
    right = (under_test.get("model") or "").strip()
    return bool(left) and left == right


def detect_path(path: str) -> dict:
    return describe_model_path(path)
