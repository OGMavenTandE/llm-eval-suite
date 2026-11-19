"""Gate reports: one standalone HTML page and one markdown file per gate, plus a packet index.

The HTML inlines the suite's token stylesheet and fonts, so it opens and
prints the same on an air-gapped machine.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from gates import GO, INVALID, NO_GO
from gates.acceptance import GateDecision

_VERDICT_CLASS = {GO: "go", NO_GO: "nogo", INVALID: "invalid"}
_VERDICT_TEXT = {
    GO: "GO: this gate passed. The work can move to the next stage.",
    NO_GO: "NO-GO: one or more bars were missed. The work does not move forward until they are met.",
    INVALID: "INVALID: the inputs cannot support a decision. No score is reported as a pass or a fail.",
}

_EXTRA_CSS = """
.gate-page { max-width: 960px; margin: 0 auto; padding: 32px 28px 48px; font-family: var(--font); color: var(--ink); }
.gate-page h1 { font-size: 26px; margin: 4px 0 2px; }
.gate-page h2 { font-size: 17px; margin: 28px 0 8px; border-bottom: 1px solid var(--line); padding-bottom: 4px; }
.gate-kicker { color: var(--ink-3); font-size: 13px; letter-spacing: .04em; text-transform: uppercase; }
.gate-meta { color: var(--ink-2); font-size: 13px; margin-top: 6px; }
.verdict { display: block; margin: 18px 0 6px; padding: 14px 16px; border-radius: 6px; font-weight: 600; font-size: 16px; }
.verdict.go { background: var(--pass-soft); color: var(--pass); border-left: 5px solid var(--pass); }
.verdict.nogo { background: var(--fail-soft); color: var(--fail); border-left: 5px solid var(--fail); }
.verdict.invalid { background: var(--warn-soft); color: var(--warn); border-left: 5px solid var(--warn); }
.gate-page table { border-collapse: collapse; width: 100%; font-size: 13.5px; margin: 6px 0; }
.gate-page th, .gate-page td { text-align: left; padding: 6px 8px; border-bottom: 1px solid var(--line); vertical-align: top; }
.gate-page th { color: var(--ink-2); font-weight: 600; background: var(--surface-sunk); }
.gate-page td.num, .gate-page th.num { text-align: right; font-variant-numeric: tabular-nums; }
.mark { font-weight: 700; white-space: nowrap; }
.mark.ok { color: var(--pass); } .mark.bad { color: var(--fail); } .mark.void { color: var(--warn); }
.gate-page ul { padding-left: 20px; } .gate-page li { margin: 4px 0; }
.gate-foot { margin-top: 32px; color: var(--ink-3); font-size: 12px; }
.gate-page code { font-size: 12px; }
@media print { .gate-page { padding: 0; } }
"""


def file_digest(path: str | Path) -> dict:
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {"file": path.name, "sha256": digest}


def _pct(v) -> str:
    return "n/a" if v is None else f"{v * 100:.1f}%"


def _mark(check: dict) -> str:
    if check["passed"]:
        return '<span class="mark ok">PASS</span>'
    if check.get("blocks_validity"):
        return '<span class="mark void">INVALID</span>'
    return '<span class="mark bad">FAIL</span>'


def _table(headers: list[str], rows: list[list[str]], numeric: set[int] = frozenset()) -> str:
    head = "".join(f'<th class="{"num" if i in numeric else ""}">{escape(h)}</th>' for i, h in enumerate(headers))
    body = "".join(
        "<tr>" + "".join(f'<td class="{"num" if i in numeric else ""}">{escape(str(c))}</td>' for i, c in enumerate(r)) + "</tr>"
        for r in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _class_rows(run: dict) -> list[list[str]]:
    return [
        [label, c["support"], c["true_positive"], _pct(c["recall"]), _pct(c["precision"]),
         "n/a" if c["f1"] is None else f"{c['f1']:.3f}"]
        for label, c in run["classes"].items()
    ]


def _run_section(title: str, run: dict) -> str:
    low, high = run["accuracy_ci95"]
    summary = (
        f"<p>{run['correct']} of {run['total']} correct ({_pct(run['accuracy'])}, 95% interval {_pct(low)} to "
        f"{_pct(high)}). {run['parsed']} of {run['total']} replies parsed to a label. Macro F1 {run['macro_f1']:.3f}.</p>"
    )
    return (
        f"<h2>{escape(title)}</h2>{summary}"
        + _table(["Label", "Items", "Found", "Recall", "Precision", "F1"], _class_rows(run), {1, 2, 3, 4, 5})
    )


def _metrics_html(d: dict) -> str:
    m = d["metrics"]
    parts = []
    gate = d["gate"]
    if gate in (2, 5):
        if "candidate" in m:
            parts.append(_run_section(f"Candidate: {m['candidate']['name']}", m["candidate"]))
        parts.append(_run_section(f"Baseline: {m['baseline']['name']}", m["baseline"]))
        if "paired" in m:
            p = m["paired"]
            parts.append("<h2>Same items, side by side</h2>" + _table(
                ["Both right", "Both wrong", "Only candidate right", "Only baseline right", "Accuracy difference", "95% interval"],
                [[p["both_right"], p["both_wrong"], p["candidate_only_right"], p["baseline_only_right"],
                  f"{p['delta'] * 100:+.1f} pts", f"{p['ci95'][0] * 100:+.1f} to {p['ci95'][1] * 100:+.1f} pts"]],
                {0, 1, 2, 3},
            ))
        if m.get("slices"):
            parts.append("<h2>Slices</h2>" + _table(
                ["Slice", "Items", "Candidate correct", "Baseline correct"],
                [[s["slice"], s["items"], f"{s['candidate_correct']} ({_pct(s['candidate_correct'] / s['items'])})",
                  f"{s['baseline_correct']} ({_pct(s['baseline_correct'] / s['items'])})"] for s in m["slices"]],
                {1, 2, 3},
            ))
    elif gate == 3:
        est = " (some estimated)" if m.get("tokens_estimated") else ""
        rows = [
            [w["task_type"], w["calls"], _pct(w["call_share"]), w["tokens"], _pct(w["token_share"]),
             f"{w['reasoning']['low']}/{w['reasoning']['medium']}/{w['reasoning']['high']}", w["sensitive"],
             w["cost_of_wrong"], f"{w['routes']['deterministic']}/{w['routes']['small']}/{w['routes']['large']}"]
            for w in m["workload_map"]
        ]
        parts.append("<h2>Workload map</h2>" + _table(
            ["Task type", "Calls", "Call share", f"Tokens{est}", "Token share", "Reasoning low/med/high",
             "Sensitive", "Cost of wrong", "Route code/small/large"], rows, {1, 2, 3, 4, 6}))
        rc, rt, total = m["route_calls"], m["route_tokens"], m["total_tokens"] or 1
        parts.append("<h2>Where the traffic goes</h2>" + _table(
            ["Route", "Calls", "Tokens", "Token share"],
            [[r, rc[r], rt[r], _pct(rt[r] / total)] for r in ("deterministic", "small", "large")], {1, 2, 3}))
    elif gate == 4:
        mis = m.get("misroutes") or []
        parts.append(f"<h2>Misroutes ({len(mis)})</h2>")
        if mis:
            parts.append(_table(["Case", "Expected", "Got", "Task type", "Reasoning", "Stress", "Note"],
                                [[x["id"], x["expected"], x["got"], x["task_type"], x["reasoning"],
                                  "yes" if x["stress"] else "", x["note"]] for x in mis]))
        else:
            parts.append("<p>Every case routed as expected.</p>")
    elif gate == 6:
        mix = m["label_mix"]
        parts.append("<h2>Predicted-label mix</h2>" + _table(
            ["Label", "Reference", "Current"],
            [[label, _pct(mix["reference"][label]), _pct(mix["current"][label])] for label in mix["reference"]],
            {1, 2}) + f"<p>PSI {mix['psi']:.3f}.</p>")
        if "current" in m:
            parts.append(_run_section(f"Current: {m['current']['name']}", m["current"]))
            parts.append(_run_section(f"Reference: {m['reference']['name']}", m["reference"]))
    return "".join(parts)


def render_html(decision: GateDecision | dict, *, generated_at: str | None = None) -> str:
    from llm_eval_suite.report_html import report_css

    d = decision.to_dict() if isinstance(decision, GateDecision) else decision
    when = generated_at or datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    check_rows = "".join(
        f"<tr><td>{_mark(c)}</td><td>{escape(c['name'])}</td><td>{escape(c['detail'])}</td></tr>" for c in d["checks"]
    )
    checks = (
        "<table><thead><tr><th>Result</th><th>Check</th><th>Detail</th></tr></thead>"
        f"<tbody>{check_rows}</tbody></table>"
    )
    findings = "".join(f"<li>{escape(f)}</li>" for f in d["findings"]) or "<li>None.</li>"
    inputs = d.get("inputs") or {}
    input_rows = "".join(
        f"<tr><td>{escape(role)}</td><td>{escape(info.get('file', ''))}</td><td><code>{escape(info.get('sha256', ''))}</code></td></tr>"
        for role, info in inputs.items()
    )
    inputs_html = (
        "<h2>Inputs</h2><table><thead><tr><th>Role</th><th>File</th><th>SHA-256</th></tr></thead>"
        f"<tbody>{input_rows}</tbody></table>" if input_rows else ""
    )
    title = f"Gate {d['gate']}: {d['title']}"
    notice_html = f'<div class="verdict invalid">{escape(d["notice"])}</div>' if d.get("notice") else ""
    return f"""<!DOCTYPE html>
<html lang="en" class="report-doc"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)} | {escape(d['workload'])}</title>
<style>{report_css()}{_EXTRA_CSS}</style></head>
<body><main class="gate-page">
<div class="gate-kicker">Gate report · {escape(d['workload'])}</div>
<h1>{escape(title)}</h1>
<div class="gate-meta">Generated {escape(when)} with llm-eval-suite gates</div>
{notice_html}<div class="verdict {_VERDICT_CLASS[d['verdict']]}">{escape(_VERDICT_TEXT[d['verdict']])}</div>
<h2>Checks</h2>{checks}
<h2>Findings</h2><ul>{findings}</ul>
{_metrics_html(d)}
{inputs_html}
<div class="gate-foot">Every bar on this page comes from the gate spec or config listed in the inputs. Corrections are handled one of three ways: fix it once, save it as a test case, or fix the system.</div>
</main></body></html>"""


def render_markdown(decision: GateDecision | dict) -> str:
    d = decision.to_dict() if isinstance(decision, GateDecision) else decision
    lines = [f"# Gate {d['gate']}: {d['title']}", "", f"Workload: {d['workload']}", "",
             *([f"> {d['notice']}", ""] if d.get("notice") else []),
             f"**{_VERDICT_TEXT[d['verdict']]}**", "", "## Checks", "", "| Result | Check | Detail |", "|---|---|---|"]
    for c in d["checks"]:
        mark = "PASS" if c["passed"] else ("INVALID" if c.get("blocks_validity") else "FAIL")
        lines.append(f"| {mark} | {c['name']} | {c['detail']} |")
    lines += ["", "## Findings", ""] + [f"- {f}" for f in d["findings"] or ["None."]]
    if d.get("inputs"):
        lines += ["", "## Inputs", "", "| Role | File | SHA-256 |", "|---|---|---|"]
        lines += [f"| {r} | {i.get('file', '')} | `{i.get('sha256', '')}` |" for r, i in d["inputs"].items()]
    return "\n".join(lines) + "\n"


def write_gate(decision: GateDecision, out_dir: str | Path, *, generated_at: str | None = None) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"gate{decision.gate}"
    paths = {
        "json": out / f"{stem}.json",
        "html": out / f"{stem}.html",
        "md": out / f"{stem}.md",
    }
    paths["json"].write_text(json.dumps(decision.to_dict(), indent=2), encoding="utf-8")
    paths["html"].write_text(render_html(decision, generated_at=generated_at), encoding="utf-8")
    paths["md"].write_text(render_markdown(decision), encoding="utf-8")
    return paths


def write_packet_index(out_dir: str | Path, workload: str) -> Path:
    """Summary of every gate JSON in the folder: the evidence packet for one workload."""
    out = Path(out_dir)
    rows = []
    for path in sorted(out.glob("gate*.json")):
        d = json.loads(path.read_text(encoding="utf-8"))
        rows.append(f"| Gate {d['gate']} | {d['title']} | {d['verdict']} | [{path.stem}.html]({path.stem}.html) |")
    text = "\n".join([f"# Evidence packet: {workload}", "", "| Gate | Title | Verdict | Report |", "|---|---|---|---|", *rows]) + "\n"
    index = out / "PACKET.md"
    index.write_text(text, encoding="utf-8")
    return index
