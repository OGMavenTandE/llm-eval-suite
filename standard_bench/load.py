"""Load pinned local JSONL. Verifies the manifest before reading rows."""

from __future__ import annotations

import json
from pathlib import Path

from standard_bench.manifest import verify_suite


def jsonl_lines(text: str) -> list[str]:
    """Split JSONL on ``\\n`` only.

    ``str.splitlines()`` also splits on U+0085, U+2028, and U+2029. MMLU-Pro
    question text contains U+0085, and that split raises ``Unterminated string``.
    A trailing ``\\r`` is stripped so ``\\r\\n`` files still parse.
    """
    if text.startswith("\ufeff"):
        text = text[1:]
    lines = []
    for line in text.split("\n"):
        if line.endswith("\r"):
            line = line[:-1]
        if line.strip():
            lines.append(line)
    return lines


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in jsonl_lines(path.read_text(encoding="utf-8")):
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return rows


def load_suite(dest: str | Path, suite: str) -> tuple[list[dict], dict]:
    """Return test rows and the verified manifest entry.

    GPQA rows are returned to the caller in memory. The caller must not write
    the question text out. Auxiliary files such as MMLU-Pro validation are
    not included. Load those with :func:`load_file`.
    """
    entry = verify_suite(dest, suite)
    root = Path(dest)
    rows: list[dict] = []
    for record in entry.get("files") or []:
        relative = str(record.get("path") or "")
        if Path(relative).name != "test.jsonl":
            continue
        rows.extend(read_jsonl(root / relative))
    return rows, entry


def load_file(dest: str | Path, suite: str, name: str) -> list[dict]:
    """Read one manifest file after the suite hash check."""
    verify_suite(dest, suite)
    path = Path(dest) / suite / name
    if not path.is_file():
        return []
    return read_jsonl(path)


def select_rows(rows: list[dict], limit: int | None, seed: int | None) -> list[dict]:
    """``limit`` keeps the first N rows. ``seed`` shuffles first, with that seed."""
    chosen = list(rows)
    if seed is not None:
        import random

        rng = random.Random(int(seed))
        rng.shuffle(chosen)
    if limit is not None and int(limit) >= 0:
        chosen = chosen[: int(limit)]
    return chosen
