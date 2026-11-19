"""Items, predictions, and label parsing."""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: not valid JSON ({exc.msg})") from exc
    return rows


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_items(path: str | Path, split: str | None = None) -> list[dict]:
    """Items need ``id``, ``text`` and ``label``. ``split`` filters on the item's split."""
    items = read_jsonl(path)
    seen: set[str] = set()
    for item in items:
        for key in ("id", "text", "label"):
            if key not in item:
                raise ValueError(f"item is missing {key!r}: {item}")
        if item["id"] in seen:
            raise ValueError(f"duplicate item id {item['id']!r}")
        seen.add(item["id"])
    if split:
        items = [item for item in items if item.get("split") == split]
    return items


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def parse_label(reply: str | None, labels: list[str]) -> str | None:
    """Return the one label the reply names, or None.

    A reply that names two different labels is not parsed, so a hedge such as
    "PRIORITY or IMMEDIATE" never counts as a hit.
    """
    if not reply:
        return None
    reply = re.sub(r"<think>.*?</think>", " ", reply, flags=re.DOTALL | re.IGNORECASE)
    norm_reply = f" {_norm(reply)} "
    exact = {_norm(label): label for label in labels}
    whole = norm_reply.strip()
    if whole in exact:
        return exact[whole]
    found = {label for key, label in exact.items() if f" {key} " in norm_reply}
    if len(found) == 1:
        return found.pop()
    return None


def load_predictions(path: str | Path, labels: list[str]) -> dict[str, dict]:
    """Predictions need ``id`` and either ``prediction`` (a label) or ``reply`` (raw text)."""
    out: dict[str, dict] = {}
    for row in read_jsonl(path):
        if "id" not in row:
            raise ValueError(f"prediction row is missing 'id': {row}")
        pred = row.get("prediction")
        if pred is not None and pred not in labels:
            pred = parse_label(str(pred), labels)
        if pred is None and row.get("reply") is not None:
            pred = parse_label(row["reply"], labels)
        out[row["id"]] = {**row, "prediction": pred}
    return out
