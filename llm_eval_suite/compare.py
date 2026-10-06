"""Before/after comparison of two stored runs."""

from __future__ import annotations

from collections import defaultdict


def _is_invalid(run: dict) -> bool:
    return run.get("validity") == "invalid" or run.get("status") == "invalid"


def _run_stamp(run: dict) -> str:
    return str(run.get("created_at") or run.get("completed_at") or "")


def _slot_name(run: dict) -> str:
    connection = run.get("connection") or {}
    name = connection.get("model") or connection.get("name") or run.get("run_id") or "Run"
    return str(name)


def _normalize_prompt(text) -> str:
    return " ".join(str(text or "").split()).casefold()


def _pair_key(item: dict) -> tuple:
    """Garak attempt ids are new every run, so pair those rows by probe and prompt."""
    suite = str(item.get("suite") or "")
    item_id = str(item.get("id") or "")
    if suite == "garak" or item_id.startswith("garak:"):
        return ("garak", str(item.get("probe") or ""), _normalize_prompt(item.get("prompt")))
    return ("id", item_id)


def _pair_items(left_items: list[dict], right_items: list[dict]):
    right_groups: dict[tuple, list[dict]] = defaultdict(list)
    for item in right_items:
        right_groups[_pair_key(item)].append(item)
    used: dict[tuple, int] = defaultdict(int)
    pairs = []
    left_unpaired = []
    for item in left_items:
        key = _pair_key(item)
        index = used[key]
        group = right_groups.get(key) or []
        if index < len(group):
            pairs.append((item, group[index]))
            used[key] = index + 1
        else:
            left_unpaired.append(item)
    right_unpaired = []
    for key, group in right_groups.items():
        right_unpaired.extend(group[used.get(key, 0) :])
    return pairs, left_unpaired, right_unpaired


def compare_runs(left: dict, right: dict, left_items: list[dict], right_items: list[dict]) -> dict:
    left_stamp = _run_stamp(left)
    right_stamp = _run_stamp(right)
    if left_stamp and right_stamp and right_stamp < left_stamp:
        left, right = right, left
        left_items, right_items = right_items, left_items
    left_card = {row["category"]: row for row in left.get("scorecard", {}).get("categories", [])}
    right_card = {row["category"]: row for row in right.get("scorecard", {}).get("categories", [])}
    left_invalid = _is_invalid(left)
    right_invalid = _is_invalid(right)
    categories = []
    keys = list(dict.fromkeys([*left_card.keys(), *right_card.keys()]))
    for key in keys:
        a = left_card.get(key) or {}
        b = right_card.get(key) or {}
        a_rate = None if left_invalid else a.get("pass_rate")
        b_rate = None if right_invalid else b.get("pass_rate")
        delta = None
        if not left_invalid and not right_invalid and a_rate is not None and b_rate is not None:
            delta = round(b_rate - a_rate, 4)
        categories.append(
            {
                "category": key,
                "label": b.get("label") or a.get("label") or key,
                "left_pass_rate": a_rate,
                "right_pass_rate": b_rate,
                "delta": delta,
                "left_status": a.get("status", "not_run"),
                "right_status": b.get("status", "not_run"),
            }
        )

    pairs, left_unpaired, right_unpaired = _pair_items(left_items, right_items)
    item_deltas = []
    unchanged_prompts = 0
    for item, other in pairs:
        if item.get("score") is None or other.get("score") is None:
            continue
        delta = round(float(other["score"]) - float(item["score"]), 4)
        if delta == 0:
            unchanged_prompts += 1
            continue
        item_deltas.append(
            {
                "id": item["id"],
                "prompt": item.get("prompt"),
                "category": item.get("category"),
                "left_score": item.get("score"),
                "right_score": other.get("score"),
                "delta": delta,
                "left_passed": item.get("passed"),
                "right_passed": other.get("passed"),
            }
        )
    unpaired_by_category: dict[str, int] = defaultdict(int)
    for item in left_unpaired + right_unpaired:
        unpaired_by_category[str(item.get("category") or "")] += 1
    item_deltas.sort(key=lambda row: (-abs(row["delta"]), -row["delta"], str(row.get("id") or "")))
    return {
        "left_run_id": left.get("run_id"),
        "right_run_id": right.get("run_id"),
        "left_label": _slot_name(left),
        "right_label": _slot_name(right),
        "left_created_at": _run_stamp(left),
        "right_created_at": _run_stamp(right),
        "unchanged_prompts": unchanged_prompts,
        "unpaired_prompts": len(left_unpaired) + len(right_unpaired),
        "unpaired_by_category": dict(unpaired_by_category),
        "left_invalid": left_invalid,
        "right_invalid": right_invalid,
        "categories": categories,
        "items": [] if left_invalid or right_invalid else item_deltas,
    }
