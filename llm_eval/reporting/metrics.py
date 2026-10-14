"""Shared score grouping for the summary CSV and the comparison table.

Both reporters walk the same per-sample records. Rounding stays at the
write site so the CSV cells and the printed table stay as they are.
"""

from __future__ import annotations


def group_scores(records: list) -> dict[str, dict[str, list]]:
    """Group evaluation rows by metric name, in first-seen order."""
    grouped: dict[str, dict[str, list]] = {}
    for record in records:
        for ev in record.get("evaluations", []):
            name = ev["metric_name"]
            bucket = grouped.get(name)
            if bucket is None:
                bucket = {"scores": [], "passed": []}
                grouped[name] = bucket
            bucket["scores"].append(ev["score"])
            bucket["passed"].append(ev["passed"])
    return grouped


def mean_pass_rate(scores: list, passed: list) -> tuple[float, float, int]:
    """Unrounded mean score, pass rate, and sample count."""
    count = len(scores)
    if not count:
        return 0.0, 0.0, 0
    return sum(scores) / count, sum(passed) / count, count
