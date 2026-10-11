"""Re-score a finished standard-bench run from saved responses. No model is loaded.

Code suites need the pinned dataset so the current plus tests can run. Other
suites use that dataset when it is passed, and otherwise the gold already
stored on the item. GPQA text is redacted in the run directory, so those
items keep the score they already have.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from llm_eval.models.context import strip_think_blocks

from standard_bench import SCORER_VERSION, SUITE_NAMES, SUITE_VERSION
from standard_bench.export import render_csv, render_json, rows_for_run
from standard_bench.load import jsonl_lines, load_suite
from standard_bench.manifest import HashMismatch, ManifestError
from standard_bench.reference import exclusions_for
from standard_bench.runner import _aggregate, _row_id, _score
from standard_bench.sandbox import RunLimits


class RescoreError(RuntimeError):
    pass


def scorer_commit() -> str:
    """HEAD of this checkout. Empty when git cannot answer."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[1],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.strip()


def _read_items(path: Path) -> list[dict]:
    rows = []
    for line in jsonl_lines(path.read_text(encoding="utf-8")):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _index_rows(suite: str, rows: list[dict]) -> dict[str, dict]:
    return {_row_id(suite, index, row): row for index, row in enumerate(rows)}


_NEEDS_ROW = frozenset({"ifeval", "humaneval_plus", "mbpp_plus"})


def _load_indexes(data_dir: str | Path | None, suites: set[str]) -> dict[str, dict[str, dict]]:
    if not data_dir:
        return {}
    root = Path(data_dir)
    indexes: dict[str, dict[str, dict]] = {}
    for suite in sorted(suites):
        if suite not in SUITE_NAMES:
            continue
        try:
            rows, _entry = load_suite(root, suite)
        except (ManifestError, HashMismatch) as exc:
            if suite in _NEEDS_ROW:
                raise RescoreError(str(exc)) from exc
            indexes[suite] = {}
            continue
        indexes[suite] = _index_rows(suite, rows)
    return indexes


def _private(suite: str, item: dict, row: dict | None) -> dict:
    if suite == "mmlu_pro":
        gold = ""
        if row is not None:
            gold = str(row.get("answer") or "")
        return {"gold": gold or str(item.get("expected") or "")}
    if suite == "gsm8k":
        gold = str(row.get("answer") or "") if row is not None else ""
        return {"gold": gold or str(item.get("expected") or "")}
    if suite == "math500":
        gold = str(row.get("answer") or "") if row is not None else ""
        return {"gold": gold or str(item.get("expected") or "")}
    if suite == "gpqa_diamond":
        return {"gold": str(item.get("expected") or "")}
    if suite == "ifeval":
        if row is None:
            raise RescoreError(
                f"IFEval item {item.get('id')} needs the dataset row. Pass --data or --fixtures."
            )
        return {"row": row}
    if suite in {"humaneval_plus", "mbpp_plus"}:
        if row is None:
            raise RescoreError(
                f"{suite} item {item.get('id')} needs the dataset row. Pass --data or --fixtures."
            )
        return {"row": row}
    raise RescoreError(f"No scorer for suite {suite}.")


def _answer_text(item: dict) -> str:
    raw = item.get("raw_response")
    if raw is None or raw == "":
        raw = item.get("response") or ""
    suite = str(item.get("suite") or "")
    return strip_think_blocks(str(raw), strip_edges=suite not in {"humaneval_plus", "mbpp_plus"})


def _apply_score(item: dict, scored: dict, commit: str) -> None:
    for key, value in scored.items():
        item[key] = value
    item["scorer_version"] = SCORER_VERSION
    item["scorer_commit"] = commit
    item["rescored"] = True


def rescore_run_dir(run_dir: str | Path, *, data_dir: str | Path | None = None) -> dict:
    """Score saved responses with the current scorer. Does not load a model."""
    directory = Path(run_dir)
    items_path = directory / "items.jsonl"
    run_path = directory / "run.json"
    if not items_path.is_file() or not run_path.is_file():
        raise FileNotFoundError("rescore needs items.jsonl and run.json in the run directory")
    items = _read_items(items_path)
    suites_present = {str(item.get("suite") or "") for item in items if item.get("suite")}
    indexes = _load_indexes(data_dir, suites_present)
    commit = scorer_commit()
    limits = RunLimits()
    updated = []
    for item in items:
        suite = str(item.get("suite") or "")
        if suite not in SUITE_NAMES:
            updated.append(item)
            continue
        stored = str(item.get("raw_response") or item.get("response") or "").strip()
        if item.get("redacted") or (suite == "gpqa_diamond" and not stored):
            item["scorer_version"] = SCORER_VERSION
            item["scorer_commit"] = commit
            updated.append(item)
            continue
        if item.get("prompt_over_budget"):
            item["scorer_version"] = SCORER_VERSION
            item["scorer_commit"] = commit
            updated.append(item)
            continue
        task_id = str(item.get("id") or "")
        short_id = task_id.split(":", 1)[1] if ":" in task_id else task_id
        excluded = exclusions_for(suite)
        if short_id in excluded:
            item["reference_excluded"] = True
            item["passed"] = False
            item["score"] = None
            item["scorer_version"] = SCORER_VERSION
            item["scorer_commit"] = commit
            item["rescored"] = True
            updated.append(item)
            continue
        row = indexes.get(suite, {}).get(task_id)
        private = _private(suite, item, row)
        needs_gold = suite in {"mmlu_pro", "gsm8k", "math500", "gpqa_diamond"}
        if needs_gold and row is None and not str(private.get("gold") or "").strip():
            raise RescoreError(
                f"{suite} item {item.get('id')} has no stored gold and no dataset row. Pass --data or --fixtures."
            )
        scored = _score(suite, _answer_text(item), private, limits)
        _apply_score(item, scored, commit)
        updated.append(item)

    items_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in updated),
        encoding="utf-8",
    )
    record = json.loads(run_path.read_text(encoding="utf-8"))
    record["scorer_version"] = SCORER_VERSION
    record["scorer_commit"] = commit
    by_suite: dict[str, list[dict]] = {}
    for item in updated:
        suite = str(item.get("suite") or "")
        if suite in SUITE_NAMES:
            by_suite.setdefault(suite, []).append(item)
    suite_rows = record.get("suites")
    if not isinstance(suite_rows, list):
        suite_rows = []
        record["suites"] = suite_rows
    seen = set()
    for suite_row in suite_rows:
        if not isinstance(suite_row, dict):
            continue
        name = str(suite_row.get("name") or "")
        if name not in by_suite:
            continue
        seen.add(name)
        _refresh_suite_row(suite_row, name, by_suite[name], commit)
    for name, suite_items in by_suite.items():
        if name in seen:
            continue
        suite_row = {"name": name, "metrics": {}}
        _refresh_suite_row(suite_row, name, suite_items, commit)
        suite_rows.append(suite_row)

    from llm_eval_suite.scoring import scorecard

    card = scorecard(updated)
    card["scorer_version"] = SCORER_VERSION
    card["scorer_commit"] = commit
    public = card.get("public_benchmarks")
    if isinstance(public, dict):
        public["scorer_version"] = SCORER_VERSION
        public["scorer_commit"] = commit
    record["scorecard"] = card
    run_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    csv_path = directory / "public_benchmarks.csv"
    json_path = directory / "public_benchmarks.json"
    exported = rows_for_run(directory)
    csv_path.write_text(render_csv(exported), encoding="utf-8")
    json_path.write_text(render_json(exported), encoding="utf-8")
    return {
        "csv": str(csv_path),
        "json": str(json_path),
        "items": len(updated),
        "scorer_version": SCORER_VERSION,
        "scorer_commit": commit,
        "scorecard": card,
    }


def _refresh_suite_row(suite_row: dict, name: str, suite_items: list[dict], commit: str) -> None:
    fresh = _aggregate(name, suite_items)
    metrics = dict(suite_row.get("metrics") or {})
    metrics.update(fresh)
    if name in {"humaneval_plus", "mbpp_plus"}:
        metrics["reference_exclusions"] = [
            {"task_id": task_id, "reason": reason}
            for task_id, reason in sorted(exclusions_for(name).items())
        ]
    metrics["scorer_version"] = SCORER_VERSION
    metrics["scorer_commit"] = commit
    suite_row["metrics"] = metrics
    suite_row["pass_rate"] = fresh.get("score")
    suite_row["suite_version"] = suite_row.get("suite_version") or SUITE_VERSION
