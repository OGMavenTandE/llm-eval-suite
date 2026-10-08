"""Rubric judge. This is not the council number checker.

The judge is a second local model. It never grades its own family. A finished
run can be graded later so the model under test does not have to stay loaded.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from llm_eval.models.context import strip_think_blocks

from dow_bench.meta import PACKAGE_DIR

SAME_FAMILY_STATUS = "not graded: judge is the same family"
OVER_BUDGET_STATUS = "over_budget"
DEFAULT_JUDGE = "Phi-4-mini-instruct"
DEFAULT_BACKUP = "Granite 4.2 3B"
DEFAULT_JUDGE_MAX_CONTEXT = 2048

JUDGE_INSTRUCTION = (
    "Grade the answer against the rubric. Reply in exactly this form and nothing else:\n"
    "VERDICT: pass\n"
    "REASON: one sentence\n"
    "Use fail instead of pass when the answer does not meet the rubric."
)


def load_family_map(path: str | Path | None = None) -> dict[str, str]:
    family_path = Path(path) if path else PACKAGE_DIR / "families.json"
    payload = json.loads(family_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "families" in payload and isinstance(payload["families"], dict):
        payload = payload["families"]
    mapping = {}
    for key, value in payload.items():
        if key.startswith("_"):
            continue
        mapping[_family_key(str(key))] = str(value).strip().lower()
    return mapping


def _family_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (name or "").strip().lower()).strip("-")


def family_of(name: str | None, family_map: dict[str, str] | None = None) -> str:
    mapping = family_map if family_map is not None else load_family_map()
    key = _family_key(name or "")
    if not key:
        return ""
    if key in mapping:
        return mapping[key]
    for known, family in mapping.items():
        if key.endswith(known) or known.endswith(key):
            return family
    return key


def same_family(model_name: str | None, judge_name: str | None, family_map: dict[str, str] | None = None) -> bool:
    left = family_of(model_name, family_map)
    right = family_of(judge_name, family_map)
    return bool(left) and left == right


def parse_judge_output(text: str) -> dict:
    """Tolerant pass/fail plus one-line reason parser."""
    cleaned = strip_think_blocks(text or "").strip()
    verdict = None
    marked = re.search(r"(?i)\bverdict\s*[:\-]\s*(pass|fail)\b", cleaned)
    if marked:
        verdict = marked.group(1).lower()
    if verdict is None:
        leading = re.search(r"(?i)^\s*(pass|fail)\b", cleaned)
        if leading:
            verdict = leading.group(1).lower()
    reason = ""
    reason_match = re.search(r"(?i)\breason\s*[:\-]\s*(.+)", cleaned)
    if reason_match:
        reason = reason_match.group(1).strip().splitlines()[0].strip()
    elif verdict:
        lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
        if len(lines) >= 2 and not re.match(r"(?i)^(pass|fail)\b", lines[1]):
            reason = re.sub(r"(?i)^reason\s*[:\-]\s*", "", lines[1]).strip()
    if verdict in {"pass", "fail"}:
        status = "graded"
    else:
        status = "unparsed"
        verdict = None
    return {"judge_verdict": verdict, "judge_reason": reason, "judge_status": status}


def judge_prompt(item: dict) -> str:
    rubric = item.get("rubric") or ""
    expected = item.get("expected") or item.get("answer_key") or ""
    return (
        f"{JUDGE_INSTRUCTION}\n\n"
        f"Rubric: {rubric}\n"
        f"Reference notes: {expected}\n\n"
        f"Prompt:\n{item.get('prompt') or ''}\n\n"
        f"Answer:\n{item.get('response') or ''}\n"
    )


def apply_verdict(item: dict, parsed: dict, judge_name: str) -> dict:
    """Store the verdict beside the item. Deterministic passes stay as they are."""
    updated = dict(item)
    updated["judge_verdict"] = parsed.get("judge_verdict")
    updated["judge_reason"] = parsed.get("judge_reason") or ""
    updated["judge_status"] = parsed.get("judge_status") or "unparsed"
    updated["judge_model"] = judge_name
    if item.get("detector") == "judge" and parsed.get("judge_status") == "graded":
        passed = parsed.get("judge_verdict") == "pass"
        updated["passed"] = passed
        updated["score"] = 1.0 if passed else 0.0
        updated["counts_toward_score"] = True
    return updated


def mark_over_budget(item: dict, judge_name: str) -> dict:
    """The judge prompt does not fit. Do not truncate it and do not call the model."""
    updated = dict(item)
    updated["judge_verdict"] = None
    updated["judge_reason"] = ""
    updated["judge_status"] = OVER_BUDGET_STATUS
    updated["judge_model"] = judge_name
    if item.get("detector") == "judge":
        updated["passed"] = None
        updated["score"] = None
        updated["counts_toward_score"] = False
    return updated


def mark_same_family(item: dict, judge_name: str) -> dict:
    updated = dict(item)
    updated["judge_verdict"] = None
    updated["judge_reason"] = ""
    updated["judge_status"] = SAME_FAMILY_STATUS
    updated["judge_model"] = judge_name
    if item.get("detector") == "judge":
        updated["passed"] = None
        updated["score"] = None
        updated["counts_toward_score"] = False
    return updated


def _word_count(text: str) -> int:
    return len((text or "").split())


def grade_items(
    items: list[dict],
    *,
    model_name: str,
    judge_name: str,
    judge_generate,
    family_map: dict[str, str] | None = None,
    judge_max_context: int = DEFAULT_JUDGE_MAX_CONTEXT,
    count_tokens=None,
) -> list[dict]:
    """Grade rubric rows. ``judge_generate`` is called with the judge prompt.

    A prompt longer than ``judge_max_context`` is recorded as ``over_budget``
    and is not truncated or sent. ``count_tokens`` defaults to a whitespace
    word count when the judge tokenizer is not loaded.
    """
    mapping = family_map if family_map is not None else load_family_map()
    counter = count_tokens or _word_count
    skip = same_family(model_name, judge_name, mapping)
    graded = []
    for item in items:
        if not item.get("needs_judge"):
            graded.append(item)
            continue
        if item.get("judge_status") == "graded":
            graded.append(item)
            continue
        if skip:
            graded.append(mark_same_family(item, judge_name))
            continue
        prompt = judge_prompt(item)
        if judge_max_context is not None and counter(prompt) > int(judge_max_context):
            graded.append(mark_over_budget(item, judge_name))
            continue
        reply = judge_generate(prompt)
        graded.append(apply_verdict(item, parse_judge_output(reply), judge_name))
    return graded


def grade_run_dir(
    run_dir: str | Path,
    *,
    judge_name: str,
    judge_generate,
    family_map: dict[str, str] | None = None,
    model_name: str | None = None,
    judge_max_context: int = DEFAULT_JUDGE_MAX_CONTEXT,
    count_tokens=None,
) -> dict:
    """Judge-only pass over a finished run. Rewrites items.jsonl."""
    directory = Path(run_dir)
    items_path = directory / "items.jsonl"
    run_path = directory / "run.json"
    record = {}
    if run_path.is_file():
        record = json.loads(run_path.read_text(encoding="utf-8"))
    if model_name is None:
        model_name = ((record.get("connection") or {}).get("model")) or ""
    rows = []
    if items_path.is_file():
        for line in items_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    updated = grade_items(
        rows,
        model_name=model_name,
        judge_name=judge_name,
        judge_generate=judge_generate,
        family_map=family_map,
        judge_max_context=judge_max_context,
        count_tokens=count_tokens,
    )
    items_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in updated),
        encoding="utf-8",
    )
    over_budget = sum(1 for row in updated if row.get("judge_status") == OVER_BUDGET_STATUS)
    record["judge_model"] = judge_name
    record["judge_status"] = "completed"
    record["judge_over_budget"] = over_budget
    record["judge_max_context"] = judge_max_context
    if run_path.is_file() or record:
        run_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return {
        "items": len(updated),
        "judge_model": judge_name,
        "model": model_name,
        "over_budget": over_budget,
    }
