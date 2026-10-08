"""Leaderboard CSV, the excluded-models note, and a short results note."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from dow_bench import SUITE_NAMES, SUITE_VERSION
from dow_bench.meta import EXCLUDED_NOTE, PACKAGE_DIR, exclusion_reason, match_model
from dow_bench.scorer import aggregate_suite

CSV_COLUMNS = (
    "model",
    "suite",
    "score",
    "n_items",
    "precision",
    "judge_model",
    "judge_agreement",
    "run_id",
    "suite_version",
    "params_total_b",
    "params_effective_b",
    "refusal_correctness",
    "helpfulness",
    "mc_accuracy",
    "short_answer_pass_rate",
    "abstention_rate",
    "citation_validity",
    "issuance_accuracy",
    "sycophancy_accuracy",
    "rubric_pass_rate",
    "gray_area_pass_rate",
)

SUBSCORE_COLUMNS = (
    "refusal_correctness",
    "helpfulness",
    "mc_accuracy",
    "short_answer_pass_rate",
    "abstention_rate",
    "citation_validity",
    "issuance_accuracy",
    "sycophancy_accuracy",
    "rubric_pass_rate",
    "gray_area_pass_rate",
)


def _blank(value) -> str:
    if value is None or value == "":
        return ""
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


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


def iter_run_dirs(runs_dir: str | Path) -> list[Path]:
    root = Path(runs_dir)
    if not root.is_dir():
        return []
    found = []
    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / "run.json").is_file():
            found.append(path)
    return found


def rows_for_run(run_dir: Path) -> list[dict]:
    """One CSV row per suite. Excluded models produce no rows."""
    record = _read_json(run_dir / "run.json") or {}
    connection = record.get("connection") or {}
    model_name = connection.get("model") or connection.get("name") or ""
    if exclusion_reason(model_name):
        return []
    items = _read_items(run_dir / "items.jsonl")
    present = {item.get("suite") for item in items}
    if not present.intersection(SUITE_NAMES):
        return []
    meta = match_model(model_name) or {}
    precision = record.get("precision") or connection.get("precision") or ""
    rows = []
    for suite in SUITE_NAMES:
        suite_items = [item for item in items if item.get("suite") == suite]
        if not suite_items:
            continue
        summary = aggregate_suite(suite, suite_items)
        version = suite_items[0].get("suite_version") or record.get("suite_version") or SUITE_VERSION
        row = {
            "model": model_name,
            "suite": suite,
            "score": _blank(summary.get("score")),
            "n_items": str(summary.get("n_items") or 0),
            "precision": precision or "",
            "judge_model": summary.get("judge_model") or record.get("judge_model") or "",
            "judge_agreement": record.get("judge_agreement") or "",
            "run_id": record.get("run_id") or run_dir.name,
            "suite_version": version,
            "params_total_b": meta.get("params_total_b") or "",
            "params_effective_b": meta.get("params_effective_b") or "",
        }
        for column in SUBSCORE_COLUMNS:
            row[column] = _blank(summary.get(column))
        rows.append(row)
    return rows


def render_csv(rows: list[dict]) -> str:
    from io import StringIO

    buffer = StringIO()
    writer = csv.DictWriter(buffer, fieldnames=list(CSV_COLUMNS), lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})
    return buffer.getvalue()


def collect_rows(runs_dir: str | Path) -> list[dict]:
    rows = []
    for run_dir in iter_run_dirs(runs_dir):
        rows.extend(rows_for_run(run_dir))
    return rows


def results_note(rows: list[dict]) -> str:
    lines = [
        "# Department of War bench results",
        "",
        f"Suite version {SUITE_VERSION}. One row per model and suite.",
        "judge_agreement stays blank until a filled grading sheet is scored.",
        "The headline for Lawful Order is the harmonic mean of refusal-correctness and helpfulness.",
        "",
        "## Excluded models",
        "",
        EXCLUDED_NOTE,
        "",
        "Excluded models are omitted from this CSV.",
        "",
        "## Rows",
        "",
    ]
    if not rows:
        lines.append("No included runs were found.")
    else:
        for row in rows:
            lines.append(
                f"- {row.get('model')} / {row.get('suite')}: score {row.get('score') or 'blank'}, "
                f"n={row.get('n_items')}, precision {row.get('precision') or 'blank'}, "
                f"run {row.get('run_id')}"
            )
    lines.append("")
    return "\n".join(lines)


def write_export(runs_dir: str | Path, csv_path: str | Path) -> dict:
    """Write the CSV, a copy of the excluded-models note, and a results note."""
    destination = Path(csv_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rows = collect_rows(runs_dir)
    destination.write_text(render_csv(rows), encoding="utf-8")
    note_path = destination.parent / "excluded_models.md"
    source_note = PACKAGE_DIR / "excluded_models.md"
    note_path.write_text(source_note.read_text(encoding="utf-8"), encoding="utf-8")
    results_path = destination.parent / "results_note.md"
    results_path.write_text(results_note(rows), encoding="utf-8")
    return {"csv": destination, "excluded_models": note_path, "results_note": results_path, "rows": rows}
