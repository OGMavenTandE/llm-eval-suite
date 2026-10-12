"""Stable dataset ids for public-benchmark items.

The id is the suite plus a task id, question id, or a hash of the question
text. It does not include the row's position, so a subset shuffle cannot
point a saved prediction at a different gold answer.
"""

from __future__ import annotations

import hashlib

# First match wins. GSM8K has none of these; its id is a question hash.
_NATIVE_FIELDS = ("task_id", "question_id", "id", "key", "unique_id")


class ItemIdError(RuntimeError):
    """A dataset row has no usable id, or two rows share one."""


def native_token(row: dict) -> str | None:
    """The row's own id, or None when the file does not carry one."""
    for field in _NATIVE_FIELDS:
        if field not in row:
            continue
        value = row.get(field)
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def question_text(suite: str, row: dict) -> str:
    """Text that identifies a row when the file has no id field."""
    if suite == "gsm8k":
        return str(row.get("question") or "")
    for field in ("question", "problem", "prompt"):
        value = row.get(field)
        if value not in (None, ""):
            return str(value)
    return ""


def question_token(suite: str, row: dict) -> str:
    """Hash of the question. The ``q`` prefix cannot collide with a positional index."""
    text = question_text(suite, row)
    if not text.strip():
        raise ItemIdError(f"{suite} row has no task id and no question text.")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return "q" + digest


def stable_row_id(suite: str, row: dict) -> str:
    """``suite:token``. The same row keeps this id after a shuffle or a subset."""
    token = native_token(row)
    if token is None:
        token = question_token(suite, row)
    return f"{suite}:{token}"


def index_rows(suite: str, rows: list[dict]) -> dict[str, dict]:
    """Map stable id to row. Two rows with one id raise ``ItemIdError``."""
    found: dict[str, dict] = {}
    for row in rows:
        row_id = stable_row_id(suite, row)
        if row_id in found:
            raise ItemIdError(
                f"{suite} id {row_id} is ambiguous: more than one dataset row uses it."
            )
        found[row_id] = row
    return found
