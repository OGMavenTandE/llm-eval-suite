"""Before/after comparison of two stored runs."""

from __future__ import annotations


def compare_runs(left: dict, right: dict, left_items: list[dict], right_items: list[dict]) -> dict:
    left_card = {row["category"]: row for row in left.get("scorecard", {}).get("categories", [])}
    right_card = {row["category"]: row for row in right.get("scorecard", {}).get("categories", [])}
    categories = []
    keys = list(dict.fromkeys([*left_card.keys(), *right_card.keys()]))
    for key in keys:
        a = left_card.get(key) or {}
        b = right_card.get(key) or {}
        a_rate = a.get("pass_rate")
        b_rate = b.get("pass_rate")
        delta = None
        if a_rate is not None and b_rate is not None:
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

    right_by_id = {item["id"]: item for item in right_items}
    item_deltas = []
    for item in left_items:
        other = right_by_id.get(item["id"])
        if other is None:
            continue
        if item.get("score") is None or other.get("score") is None:
            continue
        item_deltas.append(
            {
                "id": item["id"],
                "prompt": item.get("prompt"),
                "category": item.get("category"),
                "left_score": item.get("score"),
                "right_score": other.get("score"),
                "delta": round(float(other["score"]) - float(item["score"]), 4),
                "left_passed": item.get("passed"),
                "right_passed": other.get("passed"),
            }
        )
    item_deltas.sort(key=lambda row: row["delta"])
    return {
        "left_run_id": left.get("run_id"),
        "right_run_id": right.get("run_id"),
        "categories": categories,
        "items": item_deltas,
    }
