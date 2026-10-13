"""Compare quantization variants of one model against a chosen baseline.

Accuracy deltas use category pass rates and public-benchmark pass rates.
Speed and power deltas use the edge metrics recorded on each run. An invalid
run does not contribute an accuracy number. A missing power or latency cell
stays n/a and does not fail the tolerance.
"""

from __future__ import annotations

import json
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


def _rate_rows(run: dict, items: list[dict]) -> tuple[list[dict], list[dict]]:
    if _invalid(run):
        return [], []
    categories = []
    for row in (run.get("scorecard") or {}).get("categories") or []:
        rate = row.get("pass_rate")
        if row.get("status") in {"not_run", "withheld", "fixture"} or rate is None:
            continue
        categories.append(
            {
                "name": row.get("category"),
                "label": row.get("label") or row.get("category"),
                "pass_rate": float(rate),
                "sample_count": row.get("sample_count"),
            }
        )
    groups: dict[str, list[dict]] = {}
    for item in items:
        suite = str(item.get("suite") or "")
        if suite not in PUBLIC_BENCHMARK_SUITES and item.get("category") != "public_benchmark":
            continue
        if item.get("counts_toward_score", True) is False:
            continue
        groups.setdefault(suite or "public_benchmark", []).append(item)
    benchmarks = []
    for name in sorted(groups):
        rows = groups[name]
        rate = sum(1 for item in rows if item.get("passed")) / len(rows)
        benchmarks.append({"name": name, "label": name, "pass_rate": round(rate, 4), "sample_count": len(rows)})
    return categories, benchmarks


def _pair_accuracy(baseline_rows: list[dict], variant_rows: list[dict], *, tolerance: float) -> list[dict]:
    left = {row["name"]: row for row in baseline_rows}
    right = {row["name"]: row for row in variant_rows}
    paired = []
    for name in list(dict.fromkeys([*left.keys(), *right.keys()])):
        a = left.get(name)
        b = right.get(name)
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
                }
            )
            continue
        delta = numeric_delta(a["pass_rate"], b["pass_rate"], digits=4)
        if delta["status"] == "measured":
            delta["status"] = "pass" if delta["delta"] >= -float(tolerance) else "fail"
        paired.append({"name": name, "label": b.get("label") or a.get("label") or name, **delta})
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
        compared.append(
            {
                "label": row["label"],
                "model": row.get("model") or "",
                "run_id": (row.get("run") or {}).get("run_id"),
                "verdict": verdict,
                "failures": failures,
                "accuracy": accuracy,
                "speed": _metric_rows(base["run"], row["run"], _SPEED),
                "power": _metric_rows(base["run"], row["run"], _POWER),
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
                lines.append("No rows.")
                lines.append("")
                continue
            lines.append(
                _md_table(
                    ["Name", "Baseline", "Variant", "Delta", "Status"],
                    [
                        [
                            str(row.get("label") or row.get("name")),
                            _plain(row.get("baseline")),
                            _plain(row.get("variant")),
                            _fmt_delta(row),
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
    return "\n".join(lines).rstrip() + "\n"


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
                f"<td>{escape(_plain(row.get('variant')))}</td>"
                f"<td>{escape(_fmt_delta(row))}</td>"
                f"<td>{escape(str(row.get('status')))}</td>"
                "</tr>"
                for row in rows
            )
            empty = "<tr><td colspan=\"5\">No rows.</td></tr>" if not rows else body
            blocks.append(
                f"<h3>{escape(title)}</h3><table><tr><th>Name</th><th>Baseline</th><th>Variant</th><th>Delta</th><th>Status</th></tr>{empty}</table>"
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
<p>Accuracy deltas are the variant minus the baseline. A drop larger than the tolerance fails. Speed and power are reported and do not change the verdict.</p>
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
    "render_html",
    "render_markdown",
    "write_reports",
]
