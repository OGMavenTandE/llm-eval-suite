"""One-page HTML for a resolved map.

The layout follows Studio's landscape template. Fonts and the token sheet come
from ``report_html.report_css`` so a saved file does not request the network.
Print to PDF from the browser. This module does not launch a browser.
"""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape

from llm_eval.decision_map.schema import (
    Decision,
    DecisionMap,
    ResolvedDecision,
    ResolvedMap,
    contains_forbidden,
    trigger_text,
)
from llm_eval_suite.report_html import report_css

_ROWS_PER_PAGE = 6
_FACES = {
    "met": ("met", "✓", "Met"),
    "partial": ("par", "◐", "Partial"),
    "gap": ("gap", "✕", "Gap"),
    "not_assessed": ("na", "○", "Not assessed"),
}
_RISK_BARS = {"low": 1, "moderate": 2, "high": 3, "critical": 4}
_RISK_CLASS = {"low": "low", "moderate": "mod", "high": "high", "critical": "crit"}
_RISK_ROW = {"low": "r-low", "moderate": "r-mod", "high": "r-high", "critical": "r-crit"}
_RISK_LABEL = {"low": "Low", "moderate": "Moderate", "high": "High", "critical": "Critical"}

PAGE_CSS = """
@page { size: 11in 8.5in; margin: 0; }
html.report-doc, html.report-doc body.decision-map { background: #fff; color-scheme: light; }
body.decision-map {
  width: 11in;
  margin: 0;
  background: #fff;
  color: #15191F;
  font-family: "Public Sans", var(--font);
  font-size: 8.4pt;
  line-height: 1.25;
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}
body.decision-map p { margin: 0; }
body.decision-map h1 {
  font-size: 24pt;
  font-weight: 700;
  letter-spacing: -0.022em;
  color: #1C2430;
  margin: 4px 0 4px;
  line-height: 1.1;
}
body.decision-map h3 {
  font-size: 6.4pt;
  font-weight: 600;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  color: #626973;
  margin: 0 0 5px;
}
.sheet {
  width: 11in;
  min-height: 8.5in;
  height: 8.5in;
  box-sizing: border-box;
  padding: 0.22in 0.3in 0.14in;
  display: flex;
  flex-direction: column;
  break-after: page;
  page-break-after: always;
}
.sheet:last-child { break-after: auto; page-break-after: auto; }
.sheet.dense { font-size: 7.5pt; }
.sheet.dense h1 { font-size: 20pt; }
.top { display: flex; justify-content: space-between; align-items: flex-end; gap: 28px; }
.eye { font-size: 6.8pt; font-weight: 600; letter-spacing: 0.14em; text-transform: uppercase; color: #626973; }
.thesis { font-size: 11.5pt; color: #565D68; }
.illus { border: 1.5px solid #E8C766; background: #FFF4D6; color: #5E3F00; border-radius: 6px; padding: 8px 12px; font-size: 7.6pt; width: 2.75in; line-height: 1.35; }
.illus b { display: block; font-size: 8.2pt; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 1px; }
.sys { margin: 7px 0 5px; font-size: 7.2pt; color: #565D68; }
.sys b { color: #15191F; font-weight: 600; }
.flow { display: grid; grid-template-columns: 15fr 11fr 12.5fr 13fr 14.5fr 10fr 12.5fr 11.5fr; }
.flow div { font-size: 6.4pt; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; padding: 0 6px 3px; color: #626973; border-bottom: 2px solid #8FA3BD; }
.flow .q { grid-column: 1 / 4; }
.flow .e { grid-column: 4 / 7; color: #2C5282; border-color: #2C5282; }
.flow .a { grid-column: 7 / 9; color: #1C2430; border-color: #1C2430; }
.flow div:after { content: "  →"; letter-spacing: 0; }
.flow .a:after { content: ""; }
body.decision-map table { width: 100%; border-collapse: collapse; table-layout: fixed; font-size: inherit; }
body.decision-map th {
  text-align: left;
  font-size: 6.6pt;
  font-weight: 600;
  letter-spacing: 0.05em;
  text-transform: uppercase;
  color: #1C2430;
  padding: 6px 6px 5px;
  vertical-align: bottom;
  border-bottom: 1px solid #1C2430;
  white-space: normal;
  background: transparent;
}
body.decision-map th small { text-transform: none; letter-spacing: 0; font-weight: 400; }
body.decision-map th:nth-child(n+4):nth-child(-n+6),
body.decision-map td:nth-child(n+4):nth-child(-n+6),
body.decision-map tbody tr:hover td:nth-child(n+4):nth-child(-n+6) { background: #F3F6FA; }
body.decision-map tbody tr:hover td { background: transparent; }
body.decision-map td { padding: 3px 6px; vertical-align: top; border-bottom: 1px solid #E3E5E9; font-size: inherit; }
body.decision-map small { display: block; color: #565D68; font-size: 7.4pt; margin-top: 1px; }
.dec { border-left: 3px solid; }
.r-low .dec { border-left-color: #8FA3BD; }
.r-mod .dec { border-left-color: #2C5282; }
.r-high .dec { border-left-color: #8A5D00; }
.r-crit .dec { border-left-color: #A33A3A; }
.dec b { font-size: 8.9pt; font-weight: 600; display: block; line-height: 1.25; }
.id { font-weight: 700; color: #2C5282; font-size: 6.8pt; }
.risk { display: flex; align-items: center; gap: 5px; font-weight: 700; font-size: 7.2pt; text-transform: uppercase; letter-spacing: 0.05em; }
.bars { display: inline-flex; gap: 1.5px; align-items: flex-end; }
.bars i { width: 3px; background: #E3E5E9; display: block; }
.bars i:nth-child(1) { height: 4px; }
.bars i:nth-child(2) { height: 6px; }
.bars i:nth-child(3) { height: 8px; }
.bars i:nth-child(4) { height: 10px; }
.risk.low { color: #565D68; }
.risk.low i.on { background: #626973; }
.risk.mod { color: #2C5282; }
.risk.mod i.on { background: #2C5282; }
.risk.high { color: #8A5D00; }
.risk.high i.on { background: #8A5D00; }
.risk.crit { color: #A33A3A; }
.risk.crit i.on { background: #A33A3A; }
.ex { font-size: 5.4pt; font-weight: 700; color: #5E3F00; background: #FFF4D6; border: 1px solid #E8C766; border-radius: 3px; padding: 0 3px; letter-spacing: 0.05em; white-space: nowrap; vertical-align: 1px; }
.st { display: inline-flex; align-items: center; gap: 4px; border-radius: 9px; padding: 1px 7px 1px 3px; font-weight: 700; font-size: 7.2pt; border: 1px solid; white-space: nowrap; }
.ic { display: inline-block; width: 11px; height: 11px; border-radius: 50%; text-align: center; line-height: 11px; font-size: 6.5pt; color: #fff; }
.st.met { color: #2F7D4F; border-color: #9CC7AC; background: #EFF7F1; }
.st.met .ic { background: #2F7D4F; }
.st.par { color: #8A5D00; border-color: #E8C766; background: #FFF4D6; }
.st.par .ic { color: #8A5D00; font-size: 10pt; }
.st.gap { color: #A33A3A; border-color: #E2B1B1; background: #FBF0F0; }
.st.gap .ic { background: #A33A3A; font-size: 6pt; }
.st.na { color: #565D68; border-color: #C9CDD3; background: #fff; border-style: dashed; }
.st.na .ic { color: #565D68; }
.aul { display: flex; align-items: center; gap: 5px; }
.aul b { font-weight: 700; font-size: 7.6pt; white-space: nowrap; }
body.decision-map td .aul { flex-direction: column; align-items: flex-start; gap: 2px; }
.ramp { display: inline-flex; gap: 2px; }
.ramp i { width: 9px; height: 7px; border-radius: 1.5px; border: 1px solid #1C2430; background: #fff; display: block; }
.ramp i.on { background: #1C2430; }
.ramp.l0 i { border-color: #A33A3A; border-style: dashed; }
.a0 b { color: #A33A3A; }
.bot { display: grid; grid-template-columns: 1.45fr 1fr; gap: 18px; margin-top: 6px; }
.leg { display: grid; grid-template-columns: 1fr 1fr; gap: 1px 14px; font-size: 6.9pt; color: #565D68; }
.leg b { color: #15191F; font-weight: 600; }
.keys { font-size: 7pt; color: #565D68; }
.keys div { display: flex; align-items: center; flex-wrap: wrap; gap: 5px 8px; margin-bottom: 6px; }
.keys .lbl { width: 58px; font-weight: 600; color: #15191F; }
.foot { margin-top: auto; padding-top: 5px; border-top: 1px solid #E3E5E9; font-size: 6.4pt; color: #565D68; display: flex; justify-content: space-between; gap: 16px; }
.foot span:last-child { color: #626973; }
"""


def render_html(decision_map: DecisionMap, resolved: ResolvedMap) -> str:
    """Standalone HTML. Landscape, one sheet for up to seven rows."""
    by_id = {row.id: row for row in resolved.decisions}
    missing = [decision.id for decision in decision_map.decisions if decision.id not in by_id]
    if missing:
        raise ValueError(f"resolved map is missing decisions: {', '.join(missing)}")
    pairs = [(decision, by_id[decision.id]) for decision in decision_map.decisions]
    sheets = _sheets(pairs)
    body = "".join(
        _sheet(
            decision_map,
            resolved,
            chunk,
            first=index == 0,
            last=index == len(sheets) - 1,
            dense=len(pairs) == 7,
        )
        for index, chunk in enumerate(sheets)
    )
    title = "AI Eval Corp: Decision-to-Evidence Map"
    if decision_map.illustrative:
        title += " (illustrative example)"
    html = (
        "<!DOCTYPE html>\n"
        '<html lang="en" class="report-doc">\n'
        "<head>\n"
        '<meta charset="utf-8">\n'
        f"<title>{escape(title)}</title>\n"
        "<style>\n"
        f"{report_css()}\n"
        f"{PAGE_CSS}\n"
        "</style>\n"
        "</head>\n"
        '<body class="decision-map">\n'
        f"{body}"
        "</body>\n"
        "</html>\n"
    )
    if contains_forbidden(html):
        raise RuntimeError("rendered page contains wording this tool must not print")
    return html


def _sheets(pairs: list[tuple[Decision, ResolvedDecision]]) -> list[list[tuple[Decision, ResolvedDecision]]]:
    if len(pairs) <= 7:
        return [pairs]
    return [pairs[index : index + _ROWS_PER_PAGE] for index in range(0, len(pairs), _ROWS_PER_PAGE)]


def _sheet(
    decision_map: DecisionMap,
    resolved: ResolvedMap,
    pairs: list[tuple[Decision, ResolvedDecision]],
    *,
    first: bool,
    last: bool,
    dense: bool,
) -> str:
    klass = "sheet dense" if dense else "sheet"
    banner = _banner(decision_map) if first and decision_map.illustrative else ""
    legend = _legend(decision_map) if last else ""
    rows = "".join(_row(decision_map, decision, resolved_row) for decision, resolved_row in pairs)
    return (
        f'<section class="{klass}">'
        f"{_top(decision_map, banner)}"
        f"{_system_line(decision_map)}"
        '<div class="flow"><div class="q">Decision and stakes</div>'
        '<div class="e">Bar and evidence</div><div class="a">Authority and limits</div></div>'
        "<table><colgroup>"
        '<col style="width:15%"><col style="width:11%"><col style="width:12.5%">'
        '<col style="width:13%"><col style="width:14.5%"><col style="width:10%">'
        '<col style="width:12.5%"><col style="width:11.5%">'
        "</colgroup><thead><tr>"
        "<th>Decision</th><th>AI role</th><th>Risk if wrong</th>"
        f"<th>Required confidence{_chip(decision_map)}</th>"
        f"<th>Evidence required<small>(test, data, conditions)</small>{_chip(decision_map)}</th>"
        "<th>Evidence status</th>"
        "<th>Recommended authority<small>(decision authority grants)</small></th>"
        "<th>Reassessment trigger</th>"
        "</tr></thead><tbody>"
        f"{rows}</tbody></table>"
        f"{legend}"
        f"{_footer(decision_map, resolved)}"
        "</section>"
    )


def _top(decision_map: DecisionMap, banner: str) -> str:
    return (
        '<div class="top"><div><div class="eye">AI Eval Corp &nbsp;|&nbsp; Decision assurance</div>'
        "<h1>Decision-to-Evidence Map</h1>"
        '<p class="thesis">Each decision gets only the authority its test evidence earns.</p></div>'
        f"{banner}</div>"
    )


def _banner(decision_map: DecisionMap) -> str:
    if not decision_map.illustrative:
        return ""
    return (
        '<div class="illus"><b>Illustrative example</b>'
        "Fictional system, example values. Every number on this page is an "
        f"{_chip(decision_map, leading_space=False)}.</div>"
    )


def _system_line(decision_map: DecisionMap) -> str:
    system = decision_map.system
    name = system.name
    if system.fictional and "(fictional)" not in name.casefold():
        name = f"{name} (fictional)"
    summary = f", {escape(system.summary)}" if system.summary else ""
    if decision_map.illustrative:
        lead = f"Example system: <b>{escape(name)}</b>"
        guide = (
            " Start with the decision, read across to see the bar it must clear, "
            "the evidence behind it, and the authority AI Eval Corp recommends "
            "until a trigger reopens it."
        )
    else:
        lead = f"<b>{escape(name)}</b>"
        guide = (
            " Start with the decision, read across to the bar, the evidence, "
            "and the recommended authority written on this map."
        )
    period = "." if system.summary else ""
    return f'<p class="sys">{lead}{summary}{period} &nbsp;{guide}</p>'


def _row(decision_map: DecisionMap, decision: Decision, resolved: ResolvedDecision) -> str:
    owner = f"<small>Owner: {escape(decision.owner)}.</small>" if decision.owner else ""
    role = escape(decision.ai_role) if decision.ai_role else ""
    note = f"<small>{escape(decision.risk_note)}</small>" if decision.risk_note else ""
    confidence = escape(decision.required_confidence) if decision.required_confidence else ""
    evidence = _evidence_text(decision)
    triggers = escape(_triggers(decision))
    chip = _chip(decision_map)
    return (
        f'<tr class="{_RISK_ROW[decision.risk]}">'
        f'<td class="dec"><span class="id">{escape(decision.id)}</span>'
        f"<b>{escape(decision.decision)}</b>{owner}</td>"
        f"<td>{role}</td>"
        f"<td>{_risk(decision)}{note}</td>"
        f'<td class="cf">{confidence}{chip}</td>'
        f"<td>{escape(evidence)}</td>"
        f"<td>{_status(resolved)}{_reason(decision_map, resolved)}</td>"
        f"{_authority(decision)}"
        f"<td>{triggers}{chip}</td>"
        "</tr>"
    )


def _evidence_text(decision: Decision) -> str:
    labels = [item.label.strip().rstrip(".") for item in decision.evidence if item.label.strip()]
    text = "; ".join(labels)
    if text and not text.endswith("."):
        text += "."
    return text


def _triggers(decision: Decision) -> str:
    return "; ".join(trigger_text(trigger) for trigger in decision.triggers)


def _risk(decision: Decision) -> str:
    level = _RISK_BARS[decision.risk]
    bars = "".join(f'<i class="{"on" if index < level else ""}"></i>' for index in range(4))
    css = _RISK_CLASS[decision.risk]
    label = _RISK_LABEL[decision.risk]
    return f'<span class="risk {css}"><span class="bars">{bars}</span>{label}.</span>'


def _status(resolved: ResolvedDecision) -> str:
    css, icon, label = _FACES[resolved.status]
    return f'<span class="st {css}"><span class="ic">{icon}</span>{label}</span>'


def _reason(decision_map: DecisionMap, resolved: ResolvedDecision) -> str:
    text = resolved.reason.strip()
    chip = _chip(decision_map) if any(character.isdigit() for character in text) else ""
    if not text and not chip:
        return "<small></small>"
    return f"<small>{escape(text)}{chip}</small>"


def _authority(decision: Decision) -> str:
    if decision.recommended_authority:
        label = f"<b>{escape(decision.recommended_authority)}</b>"
    else:
        label = "<b>None written</b>"
    ramp = _ramp(decision.authority_level)
    extra = " a0" if decision.authority_level == 0 else ""
    limits = f"<small>{escape(decision.limits)}</small>" if decision.limits else ""
    return f'<td class="au"><div class="aul{extra}">{ramp}{label}</div>{limits}</td>'


def _ramp(level: int | None) -> str:
    if level is None:
        return ""
    cells = "".join(f'<i class="{"on" if index < level else ""}"></i>' for index in range(3))
    extra = " l0" if level == 0 else f" l{level}"
    return f'<span class="ramp{extra}">{cells}</span>'


def _chip(decision_map: DecisionMap, *, leading_space: bool = True) -> str:
    if not decision_map.illustrative:
        return ""
    space = " " if leading_space else ""
    return f'{space}<span class="ex">EXAMPLE</span>'


def _legend(decision_map: DecisionMap) -> str:
    glossary = ""
    if decision_map.glossary.strip():
        glossary = f'<div style="grid-column:1/3;margin-top:3px">{escape(decision_map.glossary.strip())}</div>'
    confidence = "Required confidence"
    if decision_map.illustrative:
        confidence += " (EXAMPLE)"
    return (
        '<div class="bot"><div><h3>Legend</h3><div class="leg">'
        "<div><b>Decision:</b> The choice a person makes, and who owns it.</div>"
        "<div><b>Evidence required:</b> Tests, data, and conditions that prove the bar.</div>"
        "<div><b>AI role:</b> What the AI contributes to that choice.</div>"
        "<div><b>Evidence status:</b> Met, partial, gap, or not assessed.</div>"
        "<div><b>Risk if wrong:</b> Consequence of a bad output, low to critical.</div>"
        "<div><b>Recommended authority:</b> Written by a person and shown as written. The decision authority grants it.</div>"
        f"<div><b>{confidence}:</b> The performance bar this use must clear.</div>"
        "<div><b>Reassessment trigger:</b> Events a person listed. This page does not decide that they have fired.</div>"
        f"{glossary}</div></div>"
        '<div class="keys"><h3>Key</h3>'
        f'<div><span class="lbl">Risk</span>{_key_risks()}</div>'
        f'<div><span class="lbl">Status</span>{_key_statuses()}</div>'
        f'<div><span class="lbl">Authority</span>{_key_authority()}</div>'
        "<div>Life-safety decisions never go above Human confirms.</div>"
        "</div></div>"
    )


def _key_risks() -> str:
    chunks = []
    for risk in ("low", "moderate", "high", "critical"):
        level = _RISK_BARS[risk]
        bars = "".join(f'<i class="{"on" if index < level else ""}"></i>' for index in range(4))
        chunks.append(
            f'<span class="risk {_RISK_CLASS[risk]}"><span class="bars">{bars}</span>{_RISK_LABEL[risk]}</span>'
        )
    return "".join(chunks)


def _key_statuses() -> str:
    chunks = []
    for status in ("met", "partial", "gap", "not_assessed"):
        css, icon, label = _FACES[status]
        chunks.append(f'<span class="st {css}"><span class="ic">{icon}</span>{label}</span>')
    return "".join(chunks)


def _key_authority() -> str:
    rows = (
        (0, "Not authorized"),
        (1, "Advisory only"),
        (2, "Human confirms"),
        (3, "Human on the loop"),
    )
    chunks = []
    for level, label in rows:
        extra = " a0" if level == 0 else ""
        chunks.append(f'<span class="aul{extra}">{_ramp(level)}<b>{label}</b></span>')
    return "".join(chunks)


def _footer(decision_map: DecisionMap, resolved: ResolvedMap) -> str:
    left = [resolved.caveat, "Point metrics such as latency have no interval."]
    if decision_map.illustrative:
        left.append("Fictional system. Values, thresholds, and dates are EXAMPLES and describe no fielded system.")
    left.append("Recommended authority is text from the map. The accountable official decides.")
    if decision_map.illustrative:
        left.append("AI Eval Corp, 8121 Rondelay Ln, Fairfax Station, VA 22039, scott@aieval.org")
    return (
        '<div class="foot"><span>'
        + escape(" ".join(left))
        + "</span><span>"
        + escape(_provenance(decision_map, resolved))
        + "</span></div>"
    )


def _provenance(decision_map: DecisionMap, resolved: ResolvedMap) -> str:
    generated = _iso(resolved.generated_at)
    system = decision_map.system
    version = f" {system.version}" if system.version else ""
    if not resolved.runs:
        return f"System {system.name}{version} | no suite run | generated {generated}"
    parts = []
    for run in resolved.runs:
        model = run.model or "model not recorded"
        digest = run.dataset_sha256 or "dataset hash not recorded"
        parts.append(f"Run {run.run_id} | model {model} | dataset {digest}")
    return f"System {system.name}{version} | " + " ; ".join(parts) + f" | generated {generated}"


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
