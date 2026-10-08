"""Suite execution shared by the three Department of War runners."""

from __future__ import annotations

from llm_eval.models.context import strip_think_blocks
from dow_bench import SUITE_VERSION
from dow_bench.items import load_items, resolve_item_path
from dow_bench.scorer import aggregate_suite, score_response

CATEGORY = {
    "dow_knowledge": "hallucination_factuality",
    "honest_broker": "hallucination_factuality",
    "lawful_order": "robustness",
}

LABEL = {
    "dow_knowledge": "Department of War knowledge",
    "honest_broker": "Honest broker",
    "lawful_order": "Lawful order",
}


def _cancelled(ctx) -> bool:
    return bool(ctx.cancel is not None and ctx.cancel.is_set())


def run_dow_suite(ctx, config: dict, suite: str) -> dict:
    path, sample_note = resolve_item_path(suite, config)
    rows = load_items(path)
    cap = config.get("max_new_tokens") or ctx.connection.get("max_new_tokens") or 512
    items = []
    done = 0
    total = len(rows)
    for row in rows:
        item_id = f"{suite}:{row.get('id')}"
        if item_id in ctx.completed_ids:
            done += 1
            continue
        if _cancelled(ctx):
            return _partial(suite, items, total, done, sample_note)
        think_cap = ctx.connection.get("thinking_max_tokens")
        result = ctx.model.generate(
            str(row.get("prompt") or ""),
            max_tokens=int(cap),
            thinking_max_tokens=think_cap,
        )
        raw_response = (result.metadata or {}).get("raw_text")
        if raw_response is None:
            raw_response = result.text
        answer = strip_think_blocks(result.text)
        scored = score_response(row, answer)
        item = {
            "id": item_id,
            "suite": suite,
            "category": CATEGORY[suite],
            "type": row.get("type"),
            "prompt": row.get("prompt") or "",
            "response": answer,
            "raw_response": raw_response,
            "empty": not bool(answer.strip()),
            "expected": row.get("answer_key") or "",
            "rubric": row.get("rubric") or "",
            "source": "live",
            "dataset_category": row.get("type"),
            "public_source": row.get("source") or {},
            "suite_version": SUITE_VERSION,
            "sample_note": sample_note,
            **scored,
        }
        items.append(item)
        if ctx.on_item is not None:
            ctx.on_item(item)
        done += 1
        if ctx.on_progress is not None:
            ctx.on_progress(done)
    summary = aggregate_suite(suite, items)
    notes = f"Stage-1 {LABEL[suite]} suite, version {SUITE_VERSION}."
    if sample_note:
        notes = sample_note + " " + notes
    return {
        "name": suite,
        "source": "live",
        "label": LABEL[suite],
        "notes": notes.strip(),
        "items": items,
        "done": done,
        "total": total,
        "suite_version": SUITE_VERSION,
        "subscores": summary,
    }


def _partial(suite: str, items: list[dict], total: int, done: int, sample_note: str) -> dict:
    notes = "Stopped because the run was cancelled."
    if sample_note:
        notes = sample_note + " " + notes
    return {
        "name": suite,
        "source": "live",
        "label": LABEL[suite],
        "notes": notes,
        "items": items,
        "done": done,
        "total": total,
        "cancelled": True,
        "suite_version": SUITE_VERSION,
    }
