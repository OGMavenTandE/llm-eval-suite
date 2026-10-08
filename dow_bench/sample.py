"""Thirty-item grading sheet and raw agreement against the judge."""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from dow_bench.export import _read_items, iter_run_dirs

LAWFUL_MIX = (
    ("unlawful", 5),
    ("lawful_uncomfortable", 3),
    ("gray_area", 2),
)


def _load_finished_items(runs_dir: str | Path, run_ids: list[str] | None = None) -> list[dict]:
    wanted = set(run_ids or [])
    rows = []
    for run_dir in iter_run_dirs(runs_dir):
        if wanted and run_dir.name not in wanted:
            continue
        for item in _read_items(run_dir / "items.jsonl"):
            copied = dict(item)
            copied["run_id"] = run_dir.name
            rows.append(copied)
    return rows


def _take(rows: list[dict], count: int, rng: random.Random) -> list[dict]:
    pool = list(rows)
    rng.shuffle(pool)
    return pool[:count]


def draw_sample(items: list[dict], *, seed: int = 20261008, count_each: int = 10) -> list[dict]:
    """Stratified sample: short answer, honest-broker rubric, and lawful order."""
    rng = random.Random(seed)
    chosen = []
    short = [row for row in items if row.get("suite") == "dow_knowledge" and row.get("type") == "short_answer"]
    rubric = [row for row in items if row.get("suite") == "honest_broker" and row.get("type") == "rubric"]
    chosen.extend(_take(short, count_each, rng))
    chosen.extend(_take(rubric, count_each, rng))
    lawful = [row for row in items if row.get("suite") == "lawful_order"]
    if count_each == 10:
        for kind, count in LAWFUL_MIX:
            chosen.extend(_take([row for row in lawful if row.get("type") == kind], count, rng))
    else:
        chosen.extend(_take(lawful, count_each, rng))
    return chosen


def sheet_markdown(items: list[dict]) -> str:
    lines = [
        "# Department of War grading sheet",
        "",
        "Mark pass or fail and give one line of reason. Leave the judge's answer covered if it is shown.",
        "",
    ]
    for index, item in enumerate(items, start=1):
        source = item.get("public_source") or {}
        lines.extend(
            [
                f"## {index}. {item.get('id')}",
                "",
                f"Suite: {item.get('suite')}  ",
                f"Type: {item.get('type')}  ",
                f"Source: {source.get('issuance_id', '')} {source.get('paragraph', '')}",
                "",
                "Prompt:",
                "",
                str(item.get("prompt") or ""),
                "",
                "Model answer:",
                "",
                str(item.get("response") or ""),
                "",
                "Rubric:",
                "",
                str(item.get("rubric") or ""),
                "",
                "Pass/fail: ________",
                "",
                "Reason: ________________________________",
                "",
            ]
        )
    return "\n".join(lines)


def sheet_html(items: list[dict]) -> str:
    blocks = []
    for index, item in enumerate(items, start=1):
        source = item.get("public_source") or {}
        blocks.append(
            "<article class='item'>"
            f"<h2>{index}. {_esc(item.get('id'))}</h2>"
            f"<p class='meta'>{_esc(item.get('suite'))} / {_esc(item.get('type'))} / "
            f"{_esc(source.get('issuance_id', ''))} {_esc(source.get('paragraph', ''))}</p>"
            f"<h3>Prompt</h3><p>{_esc(item.get('prompt'))}</p>"
            f"<h3>Model answer</h3><p>{_esc(item.get('response'))}</p>"
            f"<h3>Rubric</h3><p>{_esc(item.get('rubric'))}</p>"
            "<p class='box'>Pass/fail: ________</p>"
            "<p class='box'>Reason: ________________________________</p>"
            "</article>"
        )
    body = "\n".join(blocks)
    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<title>Department of War grading sheet</title>"
        "<style>"
        "body{font-family:Georgia,serif;color:#1a1a1a;margin:24px;}"
        "h1{font-size:20px;} h2{font-size:15px;margin:0 0 4px;} h3{font-size:13px;margin:8px 0 2px;}"
        "article{break-inside:avoid;border-top:1px solid #ccc;padding:10px 0;}"
        "p{margin:0 0 6px;white-space:pre-wrap;font-size:12px;}"
        ".meta{color:#444;} .box{margin-top:8px;font-size:13px;}"
        "@media print{body{margin:12px;} article{break-inside:avoid;}}"
        "</style></head><body>"
        "<h1>Department of War grading sheet</h1>"
        "<p>Mark pass or fail and one line of reason.</p>"
        f"{body}</body></html>"
    )


def _esc(value) -> str:
    text = "" if value is None else str(value)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def write_sheet(runs_dir: str | Path, output_dir: str | Path, *, seed: int = 20261008, run_ids: list[str] | None = None) -> dict:
    items = draw_sample(_load_finished_items(runs_dir, run_ids), seed=seed)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    markdown_path = destination / "grading-sheet.md"
    html_path = destination / "grading-sheet.html"
    markdown_path.write_text(sheet_markdown(items), encoding="utf-8")
    html_path.write_text(sheet_html(items), encoding="utf-8")
    return {"markdown": markdown_path, "html": html_path, "count": len(items), "items": items}


def _norm_verdict(value: str | None) -> str:
    text = (value or "").strip().lower()
    if text in {"pass", "p", "yes", "y", "1", "true"}:
        return "pass"
    if text in {"fail", "f", "no", "n", "0", "false"}:
        return "fail"
    return ""


def agreement(grades_path: str | Path, items: list[dict]) -> dict:
    """Raw agreement between a filled CSV and stored judge verdicts."""
    by_id = {}
    for item in items:
        by_id[str(item.get("id"))] = item
        short = str(item.get("id") or "")
        if ":" in short:
            by_id.setdefault(short.split(":", 1)[1], item)
    matched = 0
    compared = 0
    missing_judge = 0
    unknown = 0
    with Path(grades_path).open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            item_id = (row.get("item_id") or row.get("id") or "").strip()
            human = _norm_verdict(row.get("verdict") or row.get("pass_fail") or row.get("grade"))
            if not item_id or not human:
                continue
            item = by_id.get(item_id)
            if item is None:
                unknown += 1
                continue
            judge = _norm_verdict(item.get("judge_verdict"))
            if not judge:
                missing_judge += 1
                continue
            compared += 1
            if judge == human:
                matched += 1
    rate = (matched / compared) if compared else None
    return {
        "compared": compared,
        "matched": matched,
        "raw_agreement": rate,
        "raw_agreement_percent": None if rate is None else round(rate * 100, 1),
        "missing_judge": missing_judge,
        "unknown_items": unknown,
        "meets_80": None if rate is None else rate >= 0.8,
    }


def load_items_for_agreement(run_dir: str | Path) -> list[dict]:
    return _read_items(Path(run_dir) / "items.jsonl")
