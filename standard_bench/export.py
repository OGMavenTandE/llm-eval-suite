"""CSV and JSON export for public benchmarks.

Rows are suite metrics. GPQA question text is not a column. Every row sets
counts_toward_dow_score to false.
"""

from __future__ import annotations

import csv
import json
from io import StringIO
from pathlib import Path

from standard_bench import SUITE_NAMES, SUITE_VERSION
from standard_bench.runner import _aggregate

CSV_COLUMNS = (
    "model",
    "suite",
    "metric",
    "score",
    "n_items",
    "n_scored",
    "precision",
    "run_id",
    "suite_version",
    "full_size",
    "limit",
    "prompt_budget",
    "answer_cap",
    "thinking_budget",
    "counts_toward_dow_score",
)

METRIC_KEYS = {
    "mmlu_pro": ("accuracy",),
    "gpqa_diamond": ("accuracy",),
    "gsm8k": ("accuracy",),
    "math500": ("accuracy",),
    "ifeval": ("prompt_strict", "prompt_loose", "instruction_strict", "instruction_loose"),
    "humaneval_plus": ("pass_at_1_base", "pass_at_1_plus"),
    "mbpp_plus": ("pass_at_1_base", "pass_at_1_plus"),
}


def _blank(value) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _read_items(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def metric_rows(metrics: dict, suite: str) -> list[tuple[str, object]]:
    keys = METRIC_KEYS.get(suite, ("score",))
    found = []
    for key in keys:
        if key in metrics:
            found.append((key, metrics.get(key)))
    if not found and metrics.get("score") is not None:
        found.append(("score", metrics.get("score")))
    return found


def rows_for_run(run_dir: Path) -> list[dict]:
    record = _read_json(run_dir / "run.json")
    connection = record.get("connection") or {}
    model_name = connection.get("model") or connection.get("name") or ""
    precision = record.get("precision") or connection.get("precision") or ""
    items = _read_items(run_dir / "items.jsonl")
    suite_metrics = {}
    for suite_row in record.get("suites") or []:
        if suite_row.get("name") in SUITE_NAMES and isinstance(suite_row.get("metrics"), dict):
            suite_metrics[suite_row["name"]] = suite_row["metrics"]
    rows = []
    for suite in SUITE_NAMES:
        suite_items = [item for item in items if item.get("suite") == suite]
        if not suite_items and suite not in suite_metrics:
            continue
        metrics = suite_metrics.get(suite) or _aggregate(suite, suite_items)
        for metric, score in metric_rows(metrics, suite):
            rows.append(
                {
                    "model": model_name,
                    "suite": suite,
                    "metric": metric,
                    "score": _blank(score),
                    "n_items": _blank(metrics.get("n_items") if metrics.get("n_items") is not None else len(suite_items)),
                    "n_scored": _blank(metrics.get("n_scored")),
                    "precision": precision,
                    "run_id": record.get("run_id") or run_dir.name,
                    "suite_version": record.get("suite_version") or SUITE_VERSION,
                    "full_size": _blank(metrics.get("full_size")),
                    "limit": _blank(metrics.get("limit")),
                    "prompt_budget": _blank(metrics.get("prompt_budget") or record.get("prompt_budget")),
                    "answer_cap": _blank(metrics.get("answer_cap") or record.get("answer_cap")),
                    "thinking_budget": _blank(metrics.get("thinking_budget") or record.get("thinking_budget")),
                    "counts_toward_dow_score": "false",
                }
            )
    return rows


def render_csv(rows: list[dict]) -> str:
    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})
    return buffer.getvalue()


def render_json(rows: list[dict]) -> str:
    payload = {
        "suite_version": SUITE_VERSION,
        "counts_toward_dow_score": False,
        "rows": rows,
    }
    return json.dumps(payload, indent=2) + "\n"


def iter_run_dirs(runs_dir: str | Path) -> list[Path]:
    root = Path(runs_dir)
    if not root.is_dir():
        return []
    return [path for path in sorted(root.iterdir()) if path.is_dir() and (path / "run.json").is_file()]


def collect_rows(runs_dir: str | Path) -> list[dict]:
    rows = []
    for run_dir in iter_run_dirs(runs_dir):
        rows.extend(rows_for_run(run_dir))
    return rows


def write_export(runs_dir: str | Path, csv_path: str | Path) -> dict:
    destination = Path(csv_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(runs_dir)
    destination.write_text(render_csv(rows), encoding="utf-8")
    json_path = destination.with_suffix(".json")
    json_path.write_text(render_json(rows), encoding="utf-8")
    return {"csv": destination, "json": json_path, "rows": rows}
