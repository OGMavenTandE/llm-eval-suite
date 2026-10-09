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


def _answer_cap(ctx, config: dict) -> int:
    """Preset cap, unless the caller set ``--max-new-tokens`` on this run."""
    if ctx.connection.get("max_new_tokens_explicit"):
        return int(ctx.connection.get("max_new_tokens") or 1024)
    if config.get("max_new_tokens"):
        return int(config["max_new_tokens"])
    if ctx.connection.get("max_new_tokens"):
        return int(ctx.connection["max_new_tokens"])
    return 1024


def _hit_token_cap(result, cap: int) -> bool:
    meta = dict(getattr(result, "metadata", None) or {})
    if "hit_token_cap" in meta:
        return bool(meta.get("hit_token_cap"))
    completion = meta.get("completion_tokens")
    if completion is None:
        completion = getattr(result, "tokens_used", None)
    if completion is None:
        return False
    return int(completion) >= int(cap)


def run_dow_suite(ctx, config: dict, suite: str) -> dict:
    path, sample_note = resolve_item_path(suite, config)
    rows = load_items(path)
    cap = _answer_cap(ctx, config)
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
        generate_kwargs = {
            "max_tokens": int(cap),
            "thinking_max_tokens": think_cap,
        }
        if ctx.connection.get("prompt_budget") is not None:
            generate_kwargs["prompt_budget"] = int(ctx.connection["prompt_budget"])
        result = ctx.model.generate(str(row.get("prompt") or ""), **generate_kwargs)
        meta = dict(getattr(result, "metadata", None) or {})
        raw_response = meta.get("raw_text")
        if raw_response is None:
            raw_response = result.text
        answer = strip_think_blocks(result.text)
        over_budget = bool(meta.get("prompt_over_budget"))
        scored = score_response(row, answer)
        item = {
            "id": item_id,
            "suite": suite,
            "category": CATEGORY[suite],
            "type": row.get("type"),
            "prompt": row.get("prompt") or "",
            "response": answer,
            "raw_response": raw_response,
            "empty": not bool(str(answer or "").strip()),
            "expected": row.get("answer_key") or "",
            "answer_key": row.get("answer_key") or "",
            "expected_ids": list(row.get("expected_ids") or []),
            "correction_phrases": list(row.get("correction_phrases") or []),
            "hit_token_cap": False if over_budget else _hit_token_cap(result, cap),
            "think_truncated": bool(meta.get("think_truncated")),
            "stopped_on_newline_run": bool(meta.get("stopped_on_newline_run")),
            "prompt_over_budget": over_budget,
            "prompt_budget_error": meta.get("prompt_budget_error") or "",
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
