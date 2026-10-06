import csv
import json
from pathlib import Path

REQUIRED_FIELDS = {"prompt", "expected_answer"}
OPTIONAL_FIELDS = {"category", "difficulty", "metadata", "source"}


def load_dataset(path: str) -> list[dict]:
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    suffix = p.suffix.lower()
    if suffix == ".jsonl":
        rows = _load_jsonl(p)
    elif suffix == ".csv":
        rows = _load_csv(p)
    else:
        raise ValueError(f"Unsupported file format '{suffix}'. Expected .jsonl or .csv.")

    rows = [_alias_expected_answer(row) for row in rows]
    _validate(rows, path)
    return rows


def _alias_expected_answer(row: dict) -> dict:
    """Accept ``expected`` and ``answer`` as names for ``expected_answer``."""
    if not isinstance(row, dict):
        return row
    current = row.get("expected_answer")
    if isinstance(current, str) and current.strip():
        return row
    for key in ("expected", "answer"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            copied = dict(row)
            copied["expected_answer"] = value
            return copied
    return row


def _load_jsonl(p: Path) -> list[dict]:
    rows = []
    with p.open(encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON on line {line_num} of {p}: {e}") from e
    return rows


def _load_csv(p: Path) -> list[dict]:
    rows = []
    with p.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_num, row in enumerate(reader, start=2):  # 2 = first data row after header
            rows.append(dict(row))
    return rows


def _validate(rows: list[dict], path: str):
    if not rows:
        raise ValueError(f"Dataset is empty: {path}")

    for i, row in enumerate(rows):
        missing = REQUIRED_FIELDS - row.keys()
        if missing:
            raise ValueError(
                f"Row {i + 1} in '{path}' is missing required field(s): {sorted(missing)}. "
                "Each row must have prompt and expected_answer. "
                "The fields expected and answer are accepted aliases for expected_answer."
            )
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError(f"Row {i + 1} in '{path}': 'prompt' must be a non-empty string.")
        if not isinstance(row.get("expected_answer"), str) or not row["expected_answer"].strip():
            raise ValueError(
                f"Row {i + 1} in '{path}': 'expected_answer' must be a non-empty string."
            )
