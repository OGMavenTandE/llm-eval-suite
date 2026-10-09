"""Recompute deterministic scores from a finished run. No model is loaded."""

from __future__ import annotations

import json
from pathlib import Path

from dow_bench import SUITE_NAMES
from dow_bench.export import render_csv, rows_for_run
from dow_bench.scorer import aggregate_suite, rescore_item


def _read_items(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict):
            rows.append(row)
    return rows


def rescore_run_dir(run_dir: str | Path) -> dict:
    """Score saved responses again, keep judge verdicts, rewrite the CSV and summary."""
    directory = Path(run_dir)
    items_path = directory / "items.jsonl"
    run_path = directory / "run.json"
    if not items_path.is_file() or not run_path.is_file():
        raise FileNotFoundError("rescore needs items.jsonl and run.json in the run directory")
    record_preview = json.loads(run_path.read_text(encoding="utf-8"))
    connection = record_preview.get("connection") or {}
    model_name = connection.get("model") or connection.get("name") or ""
    updated = [rescore_item(item, model_name=model_name) for item in _read_items(items_path)]
    items_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in updated),
        encoding="utf-8",
    )
    record = json.loads(run_path.read_text(encoding="utf-8"))
    summaries = {}
    for suite in SUITE_NAMES:
        rows = [item for item in updated if item.get("suite") == suite]
        if rows:
            summaries[suite] = aggregate_suite(suite, rows)
    record["summary"] = summaries
    record["hit_token_cap_count"] = sum(1 for item in updated if item.get("hit_token_cap") is True)
    record["think_truncated_count"] = sum(1 for item in updated if item.get("think_truncated") is True)
    record["prompt_over_budget_count"] = sum(1 for item in updated if item.get("prompt_over_budget") is True)
    record["stopped_on_newline_run_count"] = sum(
        1 for item in updated if item.get("stopped_on_newline_run") is True
    )
    record["judge_over_budget"] = sum(1 for item in updated if item.get("judge_status") == "over_budget")
    record["judge_skipped"] = sum(1 for item in updated if item.get("judge_status") == "judge_skipped")
    record["judge_same_family_fallback"] = any(
        item.get("judge_same_family_fallback") and item.get("judge_status") == "graded" for item in updated
    )
    graders = [
        str(item.get("judge_model") or "")
        for item in updated
        if item.get("judge_status") == "graded" and item.get("judge_model")
    ]
    if graders:
        record["judge_model"] = graders[0]
    from llm_eval_suite.scoring import scorecard

    record["scorecard"] = scorecard(updated)
    suites = record.get("suites")
    if isinstance(suites, list):
        for suite_row in suites:
            if isinstance(suite_row, dict) and suite_row.get("name") in summaries:
                suite_row["subscores"] = summaries[suite_row["name"]]
    run_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    csv_path = directory / "dow_leaderboard.csv"
    csv_path.write_text(render_csv(rows_for_run(directory)), encoding="utf-8")
    summary_path = directory / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2) + "\n", encoding="utf-8")
    return {
        "csv": str(csv_path),
        "summary": str(summary_path),
        "items": len(updated),
        "hit_token_cap_count": record["hit_token_cap_count"],
        "think_truncated_count": record["think_truncated_count"],
        "prompt_over_budget_count": record["prompt_over_budget_count"],
        "judge_over_budget": record["judge_over_budget"],
        "suites": summaries,
    }
