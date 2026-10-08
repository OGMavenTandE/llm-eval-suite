"""Model metadata for the stage-1 leaderboard."""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent

EXCLUDED_NOTE = (
    "These models were excluded because, although marketed at a smaller effective size, "
    "their total weights exceed the 6B category. Gemma 4 E4B is 7.996B total versus its "
    "'4.5B effective' label, and Arcee Trinity Nano is 6.120B total with 1B active (MoE)."
)


def _compact(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (value or "").lower())


@lru_cache(maxsize=1)
def load_models(path: str | None = None) -> list[dict]:
    model_path = Path(path) if path else PACKAGE_DIR / "models.json"
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    return list(payload.get("models") or [])


def included_models(path: str | None = None) -> list[dict]:
    return [row for row in load_models(path) if not row.get("excluded")]


def excluded_models(path: str | None = None) -> list[dict]:
    return [row for row in load_models(path) if row.get("excluded")]


def _keys(row: dict) -> set[str]:
    keys = {
        _compact(str(row.get("id") or "")),
        _compact(str(row.get("display_name") or "")),
        _compact(str(row.get("hf_repo_id") or "")),
    }
    for alias in row.get("aliases") or []:
        keys.add(_compact(str(alias)))
    keys.discard("")
    return keys


def match_model(name: str | None, path: str | None = None) -> dict | None:
    """Join a run's model string to models.json. Exact compact match only."""
    target = _compact(name or "")
    if not target:
        return None
    for row in load_models(path):
        if target in _keys(row):
            return row
    return None


def thinking_default(name: str | None, path: str | None = None) -> bool:
    row = match_model(name, path)
    return bool(row and row.get("thinking_default"))


def hybrid_mamba(name: str | None, path: str | None = None) -> bool:
    row = match_model(name, path)
    return bool(row and row.get("hybrid_mamba"))


def exclusion_reason(name: str | None, path: str | None = None) -> str | None:
    row = match_model(name, path)
    if row is None or not row.get("excluded"):
        return None
    specific = (row.get("exclude_reason") or "").strip()
    if specific:
        return specific
    return EXCLUDED_NOTE
