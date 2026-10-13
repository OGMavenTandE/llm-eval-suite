"""Compare quantization variants of one model against a chosen baseline.

Accuracy deltas use category pass rates and public-benchmark pass rates.
Each row carries n and the passed count. A one-item swing, or a rate whose
95% binomial intervals overlap, is marked. That mark does not change the
verdict. Speed and power deltas use the edge metrics recorded on each run.
An invalid run does not contribute an accuracy number. A missing power or
latency cell stays n/a and does not fail the tolerance.
"""

from __future__ import annotations

import json
import math
from html import escape
from pathlib import Path

from llm_eval.edge.metrics import metric_number
from llm_eval_suite.report_html import report_css
from llm_eval_suite.scoring import PUBLIC_BENCHMARK_SUITES

def _invalid(run: dict) -> bool:
    return run.get("validity") == "invalid" or run.get("status") == "invalid"


def _cell_number(node, *path: str):
    current = node
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    if path and isinstance(current, dict) and "value" in current:
        return metric_number(current)
    return metric_number(current) if isinstance(current, dict) else None


def _na_delta(baseline, variant, reason: str) -> dict:
    return {"baseline": baseline, "variant": variant, "delta": "n/a", "reason": reason, "status": "n/a"}


def numeric_delta(baseline, variant, *, digits: int = 4) -> dict:
    if not isinstance(baseline, (int, float)) or not isinstance(variant, (int, float)):
        missing = "baseline" if not isinstance(baseline, (int, float)) else "variant"
        return _na_delta(baseline if baseline is not None else "n/a", variant if variant is not None else "n/a", f"{missing} was not measured")
    delta = round(float(variant) - float(baseline), digits)
    return {
        "baseline": baseline,
        "variant": variant,
        "delta": delta,
        "reason": "",
        "status": "measured",
    }


def wilson_interval(passed: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a binomial pass rate."""
    if n <= 0:
        return (0.0, 1.0)
    p = float(passed) / float(n)
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denom
    margin = z * math.sqrt((p * (1.0 - p) + z2 / (4.0 * n)) / n) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def _passed_from_rate(rate, n) -> int | None:
    """Recover a passed count only when the stored rate is that count over n."""
    if rate is None or n in (None, ""):
        return None
    try:
        total = int(n)
        raw = float(rate) * total
    except (TypeError, ValueError):
        return None
    if total < 0:
        return None
    nearest = int(round(raw))
    if abs(raw - nearest) > 1e-6:
        return None
    return nearest


def _passed_from_items(items: list[dict], category: str) -> int | None:
    rows = [
        item
        for item in items
        if item.get("category") == category and item.get("counts_toward_score", True) is not False
    ]
    if not rows:
        return None
    return sum(1 for item in rows if item.get("passed"))


def noise_flag(baseline: dict, variant: dict) -> tuple[bool, str]:
    """Mark a delta that is one item, or inside overlapping 95% intervals.

    The mark is a note. It does not change pass or fail.
    """
    left = baseline.get("passed")
    right = variant.get("passed")
    left_n = baseline.get("sample_count")
    right_n = variant.get("sample_count")
    if left is None or right is None or not left_n or not right_n:
        return False, ""
    item_delta = abs(int(right) - int(left))
    if item_delta <= 1:
        return True, "within +/- 1 item"
    left_lo, left_hi = wilson_interval(int(left), int(left_n))
    right_lo, right_hi = wilson_interval(int(right), int(right_n))
    if left_lo <= right_hi and right_lo <= left_hi:
        return True, "within the 95% binomial interval"
    return False, ""


def _sample_count(value) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _benchmark_from_suites(run: dict) -> list[dict]:
    """Standard-suite pass rates stored on ``run.json``.

    Those items set ``counts_toward_score`` false so they stay out of the
    DoW scorecard. The suite record is the rate the run actually saved.
    """
    rows = []
    for suite in run.get("suites") or []:
        name = str(suite.get("name") or "")
        if name not in PUBLIC_BENCHMARK_SUITES:
            continue
        if suite.get("not_run") or suite.get("status") in {"not_run", "withheld", "failed"}:
            continue
        metrics = suite.get("metrics") if isinstance(suite.get("metrics"), dict) else {}
        rate = suite.get("pass_rate")
        if rate is None:
            rate = metrics.get("score")
        if rate is None:
            continue
        try:
            rate_value = float(rate)
        except (TypeError, ValueError):
            continue
        sample_count = _sample_count(metrics.get("n_scored"))
        if sample_count is None:
            sample_count = _sample_count(metrics.get("n_items"))
        if sample_count is None:
            sample_count = _sample_count(suite.get("done"))
        rows.append(
            {
                "name": name,
                "label": suite.get("label") or name,
                "pass_rate": rate_value,
                "sample_count": sample_count,
                "passed": _passed_from_rate(rate_value, sample_count),
            }
        )
    return rows


def _benchmark_from_items(items: list[dict]) -> list[dict]:
    groups: dict[str, list[dict]] = {}
    for item in items:
        suite = str(item.get("suite") or "")
        if suite not in PUBLIC_BENCHMARK_SUITES and item.get("category") != "public_benchmark":
            continue
        groups.setdefault(suite or "public_benchmark", []).append(item)
    benchmarks = []
    for name in sorted(groups):
        rows = groups[name]
        passed = sum(1 for item in rows if item.get("passed"))
        rate = passed / len(rows)
        benchmarks.append(
            {
                "name": name,
                "label": name,
                "pass_rate": round(rate, 4),
                "sample_count": len(rows),
                "passed": passed,
            }
        )
    return benchmarks


def _rate_rows(run: dict, items: list[dict]) -> tuple[list[dict], list[dict]]:
    if _invalid(run):
        return [], []
    categories = []
    for row in (run.get("scorecard") or {}).get("categories") or []:
        rate = row.get("pass_rate")
        if row.get("status") in {"not_run", "withheld", "fixture"} or rate is None:
            continue
        name = row.get("category")
        sample_count = row.get("sample_count")
        passed = _passed_from_items(items, name)
        if passed is None:
            passed = _passed_from_rate(rate, sample_count)
        categories.append(
            {
                "name": name,
                "label": row.get("label") or name,
                "pass_rate": float(rate),
                "sample_count": sample_count,
                "passed": passed,
            }
        )
    benchmarks = _benchmark_from_suites(run)
    seen = {row["name"] for row in benchmarks}
    for row in _benchmark_from_items(items):
        if row["name"] not in seen:
            benchmarks.append(row)
    return categories, benchmarks


def _pair_accuracy(baseline_rows: list[dict], variant_rows: list[dict], *, tolerance: float) -> list[dict]:
    left = {row["name"]: row for row in baseline_rows}
    right = {row["name"]: row for row in variant_rows}
    paired = []
    for name in list(dict.fromkeys([*left.keys(), *right.keys()])):
        a = left.get(name)
        b = right.get(name)
        counts = {
            "baseline_n": None if a is None else a.get("sample_count"),
            "variant_n": None if b is None else b.get("sample_count"),
            "baseline_passed": None if a is None else a.get("passed"),
            "variant_passed": None if b is None else b.get("passed"),
            "within_noise": False,
            "note": "",
        }
        if a is None or b is None:
            paired.append(
                {
                    "name": name,
                    "label": (b or a or {}).get("label") or name,
                    **_na_delta(
                        None if a is None else a["pass_rate"],
                        None if b is None else b["pass_rate"],
                        "not run on both variants",
                    ),
                    **counts,
                }
            )
            continue
        delta = numeric_delta(a["pass_rate"], b["pass_rate"], digits=4)
        if delta["status"] == "measured":
            delta["status"] = "pass" if delta["delta"] >= -float(tolerance) else "fail"
        within, note = noise_flag(a, b)
        counts["within_noise"] = within
        counts["note"] = note
        paired.append({"name": name, "label": b.get("label") or a.get("label") or name, **delta, **counts})
    return paired


_SPEED = (
    ("ttft_ms_p50", "TTFT p50 (ms)", ("ttft_ms", "p50"), 2),
    ("ttft_ms_p95", "TTFT p95 (ms)", ("ttft_ms", "p95"), 2),
    ("output_tokens_per_second", "Output tokens/s", ("output_tokens_per_second",), 2),
    ("latency_ms_p50", "Latency p50 (ms)", ("latency_ms", "p50"), 2),
    ("latency_ms_p95", "Latency p95 (ms)", ("latency_ms", "p95"), 2),
)

_POWER = (
    ("power_watts_average", "Average watts", ("power_watts_average",), 2),
    ("power_watts_peak", "Peak watts", ("power_watts_peak",), 2),
    ("joules_per_output_token", "Joules / output token", ("joules_per_output_token",), 6),
    ("peak_vram_mib", "Peak VRAM (MiB)", ("peak_vram_mib",), 1),
    ("peak_ram_mib", "Peak RAM (MiB)", ("peak_ram_mib",), 1),
)


def _reason_along(node: dict, path: tuple[str, ...]) -> str:
    current = node
    for part in path:
        if not isinstance(current, dict):
            return ""
        if current.get("value") == "n/a" and current.get("reason"):
            return str(current["reason"])
        current = current.get(part)
    if isinstance(current, dict) and current.get("reason") and current.get("value") == "n/a":
        return str(current["reason"])
    return ""


def _metric_rows(baseline: dict, variant: dict, fields) -> list[dict]:
    left = baseline.get("edge_metrics") or {}
    right = variant.get("edge_metrics") or {}
    rows = []
    for key, label, path, digits in fields:
        base = _cell_number(left, *path)
        other = _cell_number(right, *path)
        if base is None or other is None:
            reason = _reason_along(left, path) or _reason_along(right, path)
            if not left or not right:
                reason = "edge metrics were not recorded"
            row = _na_delta(
                base if base is not None else "n/a",
                other if other is not None else "n/a",
                reason or "not measured",
            )
        else:
            row = numeric_delta(base, other, digits=digits)
        rows.append({"name": key, "label": label, **row})
    return rows


def _optional_metric(baseline: dict, variant: dict, key: str, label: str, digits: int) -> list[dict]:
    left = baseline.get("edge_metrics") or {}
    right = variant.get("edge_metrics") or {}
    if key not in left and key not in right:
        return []
    return _metric_rows(
        {"edge_metrics": {key: left.get(key)}},
        {"edge_metrics": {key: right.get(key)}},
        ((key, label, (key,), digits),),
    )


def _vram_method_label(run: dict) -> str:
    cell = (run.get("edge_metrics") or {}).get("vram_method")
    if not isinstance(cell, dict):
        return ""
    value = str(cell.get("value") or "").strip()
    if not value or value == "n/a":
        return ""
    reason = str(cell.get("reason") or "").strip()
    return f"{value} ({reason})" if reason else value


def _empty_accuracy(kind: str, variant: dict) -> str:
    accuracy = variant.get("accuracy") or {}
    if kind == "categories" and accuracy.get("benchmarks"):
        return "No category rows. The standard-suite scores are in the benchmark table."
    if kind == "benchmarks" and accuracy.get("categories"):
        return "No benchmark rows."
    return "No rows."


def _verdict(accuracy_rows: list[dict]) -> str:
    if any(row.get("status") == "fail" for row in accuracy_rows):
        return "fail"
    if any(row.get("status") == "pass" for row in accuracy_rows):
        return "pass"
    return "n/a"


def build_delta_report(variants: list[dict], *, baseline: str, tolerance: float) -> dict:
    """``variants`` entries are ``{label, model, run, items}``. ``baseline`` is a label."""
    by_label = {str(row["label"]): row for row in variants}
    if baseline not in by_label:
        known = ", ".join(by_label) or "(none)"
        raise KeyError(f"Baseline '{baseline}' is not in the variants. Known labels: {known}.")
    base = by_label[baseline]
    base_categories, base_benchmarks = _rate_rows(base["run"], base.get("items") or [])
    compared = []
    for row in variants:
        if row["label"] == baseline:
            continue
        categories, benchmarks = _rate_rows(row["run"], row.get("items") or [])
        if _invalid(base["run"]) or _invalid(row["run"]):
            accuracy = {
                "categories": [
                    {
                        "name": "score",
                        "label": "Category scores",
                        **_na_delta("n/a", "n/a", "score withheld"),
                    }
                ],
                "benchmarks": [],
            }
        else:
            accuracy = {
                "categories": _pair_accuracy(base_categories, categories, tolerance=tolerance),
                "benchmarks": _pair_accuracy(base_benchmarks, benchmarks, tolerance=tolerance),
            }
        accuracy_rows = [*accuracy["categories"], *accuracy["benchmarks"]]
        verdict = _verdict(accuracy_rows)
        failures = [
            f"{item['label']} delta {item['delta']} is below -{tolerance:g}"
            for item in accuracy_rows
            if item.get("status") == "fail"
        ]
        power = _metric_rows(base["run"], row["run"], _POWER)
        power.extend(_optional_metric(base["run"], row["run"], "baseline_vram_mib", "Baseline VRAM (MiB)", 1))
        compared.append(
            {
                "label": row["label"],
                "model": row.get("model") or "",
                "run_id": (row.get("run") or {}).get("run_id"),
                "verdict": verdict,
                "failures": failures,
                "accuracy": accuracy,
                "speed": _metric_rows(base["run"], row["run"], _SPEED),
                "power": power,
                "vram_methods": {
                    "baseline": _vram_method_label(base["run"]),
                    "variant": _vram_method_label(row["run"]),
                },
            }
        )
    verdicts = [row["verdict"] for row in compared]
    if any(item == "fail" for item in verdicts):
        overall = "fail"
    elif verdicts and all(item == "pass" for item in verdicts):
        overall = "pass"
    else:
        overall = "n/a"
    return {
        "baseline": baseline,
        "baseline_model": base.get("model") or "",
        "baseline_run_id": (base.get("run") or {}).get("run_id"),
        "tolerance": float(tolerance),
        "verdict": overall,
        "variants": compared,
    }


def _fmt_delta(row: dict) -> str:
    if row.get("delta") == "n/a" or row.get("status") == "n/a":
        reason = row.get("reason") or ""
        return f"n/a ({reason})" if reason else "n/a"
    value = row.get("delta")
    if isinstance(value, (int, float)) and value > 0:
        return f"+{value}"
    return str(value)


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |"
    rule = "| " + " | ".join("---" for _ in headers) + " |"
    body = ["| " + " | ".join(cell for cell in row) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def render_markdown(report: dict) -> str:
    lines = [
        f"# Quantization comparison ({report.get('baseline')} baseline)",
        "",
        f"Tolerance: {report.get('tolerance')}. Verdict: {report.get('verdict')}.",
        "",
        "Accuracy deltas are the variant minus the baseline. A drop larger than the tolerance fails.",
        "n is the item count. A note of within +/- 1 item, or within the 95% binomial interval, means the delta is inside sampling noise. The note does not change the verdict.",
        "Speed and power deltas are reported from the edge metrics and do not change the verdict.",
        "",
    ]
    for variant in report.get("variants") or []:
        lines.append(f"## {variant.get('label')} ({variant.get('model')})")
        lines.append("")
        lines.append(f"Verdict: {variant.get('verdict')}.")
        if variant.get("failures"):
            lines.append("")
            for failure in variant["failures"]:
                lines.append(f"- {failure}")
        lines.append("")
        for title, key in (("Accuracy by category", "categories"), ("Accuracy by benchmark", "benchmarks")):
            rows = (variant.get("accuracy") or {}).get(key) or []
            lines.append(f"### {title}")
            lines.append("")
            if not rows:
                lines.append(_empty_accuracy(key, variant))
                lines.append("")
                continue
            lines.append(
                _md_table(
                    ["Name", "Baseline", "n", "Variant", "n", "Delta", "Noise", "Status"],
                    [
                        [
                            str(row.get("label") or row.get("name")),
                            _plain(row.get("baseline")),
                            _count_text(row.get("baseline_passed"), row.get("baseline_n")),
                            _plain(row.get("variant")),
                            _count_text(row.get("variant_passed"), row.get("variant_n")),
                            _fmt_delta(row),
                            str(row.get("note") or ""),
                            str(row.get("status")),
                        ]
                        for row in rows
                    ],
                )
            )
            lines.append("")
        for title, key in (("Speed", "speed"), ("Power", "power")):
            rows = variant.get(key) or []
            lines.append(f"### {title}")
            lines.append("")
            lines.append(
                _md_table(
                    ["Metric", "Baseline", "Variant", "Delta"],
                    [
                        [
                            str(row.get("label")),
                            _plain(row.get("baseline")),
                            _plain(row.get("variant")),
                            _fmt_delta(row),
                        ]
                        for row in rows
                    ],
                )
            )
            lines.append("")
            if key == "power":
                lines.extend(_vram_method_lines(report, variant))
    return "\n".join(lines).rstrip() + "\n"


def _vram_method_lines(report: dict, variant: dict) -> list[str]:
    methods = variant.get("vram_methods") or {}
    baseline = str(methods.get("baseline") or "").strip()
    other = str(methods.get("variant") or "").strip()
    if not baseline and not other:
        return []
    return [
        f"VRAM method ({report.get('baseline')}): {baseline or 'n/a'}.",
        f"VRAM method ({variant.get('label')}): {other or 'n/a'}.",
        "",
    ]


def _count_text(passed, n) -> str:
    if n in (None, ""):
        return ""
    if passed in (None, ""):
        return str(n)
    return f"{passed}/{n}"


def _plain(value) -> str:
    if value == "n/a" or value is None:
        return "n/a"
    return str(value)


def render_html(report: dict) -> str:
    sections = []
    for variant in report.get("variants") or []:
        blocks = [f"<h2>{escape(str(variant.get('label')))} ({escape(str(variant.get('model') or ''))})</h2>"]
        blocks.append(f"<p>Verdict: {escape(str(variant.get('verdict')))}.</p>")
        if variant.get("failures"):
            items = "".join(f"<li>{escape(str(item))}</li>" for item in variant["failures"])
            blocks.append(f"<ul>{items}</ul>")
        for title, key in (("Accuracy by category", "categories"), ("Accuracy by benchmark", "benchmarks")):
            rows = (variant.get("accuracy") or {}).get(key) or []
            body = "".join(
                "<tr>"
                f"<td>{escape(str(row.get('label') or row.get('name')))}</td>"
                f"<td>{escape(_plain(row.get('baseline')))}</td>"
                f"<td>{escape(_count_text(row.get('baseline_passed'), row.get('baseline_n')))}</td>"
                f"<td>{escape(_plain(row.get('variant')))}</td>"
                f"<td>{escape(_count_text(row.get('variant_passed'), row.get('variant_n')))}</td>"
                f"<td>{escape(_fmt_delta(row))}</td>"
                f"<td>{escape(str(row.get('note') or ''))}</td>"
                f"<td>{escape(str(row.get('status')))}</td>"
                "</tr>"
                for row in rows
            )
            empty = f"<tr><td colspan=\"8\">{escape(_empty_accuracy(key, variant))}</td></tr>" if not rows else body
            blocks.append(
                "<h3>"
                + escape(title)
                + "</h3><table><tr><th>Name</th><th>Baseline</th><th>n</th><th>Variant</th><th>n</th><th>Delta</th><th>Noise</th><th>Status</th></tr>"
                + empty
                + "</table>"
            )
        for title, key in (("Speed", "speed"), ("Power", "power")):
            rows = variant.get(key) or []
            body = "".join(
                "<tr>"
                f"<td>{escape(str(row.get('label')))}</td>"
                f"<td>{escape(_plain(row.get('baseline')))}</td>"
                f"<td>{escape(_plain(row.get('variant')))}</td>"
                f"<td>{escape(_fmt_delta(row))}</td>"
                "</tr>"
                for row in rows
            )
            blocks.append(
                f"<h3>{escape(title)}</h3><table><tr><th>Metric</th><th>Baseline</th><th>Variant</th><th>Delta</th></tr>{body}</table>"
            )
            if key == "power":
                for line in _vram_method_lines(report, variant):
                    text = line.strip()
                    if text:
                        blocks.append(f"<p>{escape(text)}</p>")
        sections.append("".join(blocks))
    css = report_css()
    return f"""<!DOCTYPE html>
<html lang="en" class="report-doc">
<head>
<meta charset="utf-8">
<title>Quantization comparison</title>
<style>
{css}
@media print {{
  body {{ background: white; }}
}}
</style>
</head>
<body>
<main class="report">
<h1>Quantization comparison</h1>
<p>Baseline: {escape(str(report.get('baseline')))} ({escape(str(report.get('baseline_model') or ''))}).</p>
<p>Tolerance: {escape(str(report.get('tolerance')))}. Verdict: {escape(str(report.get('verdict')))}.</p>
<p>Accuracy deltas are the variant minus the baseline. A drop larger than the tolerance fails. n is the item count. A noise note means the delta is within +/- 1 item or inside the 95% binomial interval, and that note does not change the verdict. Speed and power are reported and do not change the verdict.</p>
{''.join(sections)}
</main>
</body>
</html>
"""


def write_reports(report: dict, directory: str | Path) -> dict:
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    paths = {
        "json": folder / "report.json",
        "markdown": folder / "report.md",
        "html": folder / "report.html",
    }
    paths["json"].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    paths["markdown"].write_text(render_markdown(report), encoding="utf-8")
    paths["html"].write_text(render_html(report), encoding="utf-8")
    return {key: str(path) for key, path in paths.items()}


def edge_metric_deltas(left: dict, right: dict) -> list[dict]:
    """Later-minus-earlier edge deltas for the Compare view. Empty when neither run recorded them."""
    if not left.get("edge_metrics") and not right.get("edge_metrics"):
        return []
    rows = _metric_rows(left, right, (*_SPEED, *_POWER))
    # _metric_rows treats the first run as baseline. Compare already ordered earlier, later.
    return rows


__all__ = [
    "build_delta_report",
    "edge_metric_deltas",
    "noise_flag",
    "render_html",
    "render_markdown",
    "wilson_interval",
    "write_reports",
]
