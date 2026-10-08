"""JSONL item files for the three Department of War suites."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dow_bench import CANARY, SUITE_NAMES

PACKAGE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = PACKAGE_DIR / "public"
REPO_ROOT = PACKAGE_DIR.parent

PUBLIC_SAMPLE_NOTE = (
    "Using the committed public sample because private/{suite}.test.jsonl was not found. "
    "This is the public sample, not the held-out test set."
)


def repo_root() -> Path:
    return REPO_ROOT


def private_test_path(suite: str) -> Path:
    return repo_root() / "private" / f"{suite}.test.jsonl"


def public_sample_path(suite: str) -> Path:
    return PUBLIC_DIR / f"{suite}.public.jsonl"


def resolve_item_path(suite: str, config: dict | None = None) -> tuple[Path, str]:
    """Configured path, then private/<suite>.test.jsonl, then the public sample."""
    if suite not in SUITE_NAMES:
        raise KeyError(f"Unknown Department of War suite: {suite}")
    config = config or {}
    configured = config.get("path") or os.environ.get(f"DOW_{suite.upper()}_PATH")
    if configured:
        path = Path(str(configured))
        if not path.is_file():
            raise FileNotFoundError(f"{suite} item file not found: {path}")
        return path, ""
    private = private_test_path(suite)
    if private.is_file():
        return private, ""
    public = public_sample_path(suite)
    if not public.is_file():
        raise FileNotFoundError(f"No item file for {suite}. Expected {private} or {public}.")
    return public, PUBLIC_SAMPLE_NOTE.format(suite=suite)


def load_items(path: str | Path) -> list[dict]:
    """Load suite JSONL. A leading canary record is skipped."""
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Line {line_number} of {path} is not an object.")
        if row.get("record") == "canary":
            continue
        rows.append(row)
    return rows


def count_items(suite: str, config: dict | None = None) -> int:
    try:
        path, _note = resolve_item_path(suite, config)
    except (FileNotFoundError, KeyError):
        return 0
    return len(load_items(path))


def file_has_canary_line(path: str | Path) -> bool:
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            row = json.loads(text)
        except json.JSONDecodeError:
            if CANARY in text:
                return True
            continue
        if isinstance(row, dict) and row.get("record") == "canary" and row.get("canary") == CANARY:
            return True
    return False
