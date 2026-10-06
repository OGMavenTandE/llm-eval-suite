"""Standalone HTML report. Print to PDF from the browser. Styling is the token file."""

from __future__ import annotations

import base64
import re
from html import escape
from pathlib import Path

from llm_eval_suite.scoring import pass_percent_text

TOKENS_PATH = Path(__file__).parent / "static" / "tokens.css"
FONTS_DIR = Path(__file__).parent / "static" / "fonts"


def _embed_font_urls(css: str) -> str:
    """Inline bundled fonts so a saved report does not request ``/static/fonts``."""

    def replace(match: re.Match) -> str:
        path = FONTS_DIR / match.group(1)
        if not path.is_file():
            return match.group(0)
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f'url("data:font/woff2;base64,{encoded}")'

    return re.sub(r'url\("/static/fonts/([^"]+)"\)', replace, css)


def render_report(run: dict, items: list[dict]) -> str:
    css = ""
    if TOKENS_PATH.is_file():
        css = _embed_font_urls(TOKENS_PATH.read_text(encoding="utf-8"))
    connection = run.get("connection") or {}
    dataset = run.get("dataset") or {}
    card = run.get("scorecard") or {}
    analysis = run.get("analysis") or {}
    suites = run.get("suites") or []
    failures = [
        item
        for item in items
        if not item.get("passed") and item.get("source") == "live" and item.get("counts_toward_score", True)
    ]
    fixture_failures = [
        item
        for item in items
        if not item.get("passed") and item.get("source") != "live"
    ]

    category_rows = []
    for row in card.get("categories") or []:
        if run.get("validity") == "invalid" or run.get("status") == "invalid" or row.get("status") == "withheld":
            rate = "withheld"
        elif row.get("pass_percent") is None:
            rate = "not run"
        else:
            rate = pass_percent_text(row.get("pass_percent")) or "not run"
            source = row.get("source") or ""
            if source in {"fixture", "smoke"} or row.get("status") == "fixture":
                rate = f"{rate} (fixture)"
        category_rows.append(
            "<tr>"
            f"<td>{escape(str(row.get('label') or row.get('category')))}</td>"
            f"<td>{escape(str(row.get('status')))}</td>"
            f"<td>{escape(rate)}</td>"
            f"<td>{escape(str(row.get('sample_count')))}</td>"
            "</tr>"
        )

    suite_rows = []
    for suite in suites:
        suite_rows.append(
            "<tr>"
            f"<td>{escape(str(suite.get('name')))}</td>"
            f"<td>{escape(str(suite.get('label') or suite.get('source')))}</td>"
            f"<td>{escape(str(suite.get('notes') or ''))}</td>"
            "</tr>"
        )

    failure_rows = []
    for item in failures:
        failure_rows.append(
            "<tr>"
            f"<td>{escape(str(item.get('category')))}</td>"
            f"<td>{escape(str(item.get('source')))}</td>"
            f"<td>{escape(str(item.get('prompt') or ''))}</td>"
            f"<td>{escape(str(item.get('response') or ''))}</td>"
            f"<td>{escape(str(item.get('expected') or ''))}</td>"
            f"<td>{escape(str(item.get('score')))}</td>"
            f"<td>{escape(str((item.get('evidence') or {}).get('span') or ''))}</td>"
            f"<td>{escape(str((item.get('evidence') or {}).get('excerpt') or ''))}</td>"
            "</tr>"
        )
    if not failure_rows:
        failure_rows.append("<tr><td colspan='8'>No failing prompts were stored.</td></tr>")

    evidence_rows = []
    for item in items:
        evidence = item.get("evidence") or {}
        if item.get("suite") not in {"factcheck", "garak"} and not evidence:
            continue
        notes = item.get("detector_notes") or []
        note_text = "; ".join(
            f"{row.get('name')}: {row.get('reason') or row.get('status')}"
            for row in notes
            if row.get("status") in {"skipped", "not_applicable"}
        )
        evidence_rows.append(
            "<tr>"
            f"<td>{escape(str(item.get('suite') or ''))}</td>"
            f"<td>{escape(str(item.get('score')))}</td>"
            f"<td>{escape(str(evidence.get('match') or item.get('detector') or ''))}</td>"
            f"<td>{escape(str(evidence.get('span') or ''))}</td>"
            f"<td>{escape(str(evidence.get('excerpt') or item.get('response') or ''))}</td>"
            f"<td>{escape(note_text)}</td>"
            "</tr>"
        )
    if not evidence_rows:
        evidence_rows.append("<tr><td colspan='6'>No scored prompts stored.</td></tr>")

    verdict = card.get("verdict") or "No live categories scored"
    failure_count = card.get("failure_count")
    if failure_count is None:
        failure_count = len(failures)
    fixture_note = ""
    if fixture_failures:
        fixture_note = (
            f"<p>{len(fixture_failures)} fixture or smoke failure"
            f"{'' if len(fixture_failures) == 1 else 's'} "
            "are not included in the failing-prompt count.</p>"
        )
    narrative = analysis.get("narrative") or "No analysis has been written yet."
    source_label = analysis.get("source_label") or "not run"
    validity = run.get("validity") or "ok"
    validity_reason = run.get("validity_reason") or ""
    garak_wording = run.get("garak_wording") or ""
    security = next(
        (row for row in (card.get("categories") or []) if row.get("category") == "security_jailbreak"),
        None,
    )
    invalid_score = validity == "invalid" or run.get("status") == "invalid" or (security or {}).get("status") == "withheld"
    security_live = bool(security) and security.get("source") == "live" and security.get("pass_percent") is not None
    if invalid_score:
        garak_rate_text = "withheld"
        garak_asr_text = "withheld"
    elif security_live:
        # Same one-decimal text as the Security row. Not a second average.
        garak_rate_text = pass_percent_text(security.get("pass_percent")) or "not available"
        asr_percent = round(100 - float(security["pass_percent"]), 1)
        garak_asr_text = f"{asr_percent:.1f}%"
    else:
        garak_rate_text = "not available"
        garak_asr_text = "not available"
    invalid_banner = ""
    if validity == "invalid":
        invalid_banner = (
            "<div class='banner banner-fail'>INVALID run. "
            f"{escape(validity_reason or 'Too many empty generations.')}</div>"
        )

    return f"""<!DOCTYPE html>
<html lang="en" class="report-doc">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Evaluation report {escape(str(run.get('run_id') or ''))}</title>
<style>
{css}
@media print {{
  body {{ background: white; }}
  a {{ color: black; text-decoration: none; }}
}}
</style>
</head>
<body>
<main class="report">
<h1>Evaluation report</h1>
<p class="muted">Run {escape(str(run.get('run_id') or ''))}</p>
{invalid_banner}

<h2>Test plan</h2>
<table>
<tr><th>Model</th><td>{escape(str(connection.get('model') or ''))}</td></tr>
<tr><th>Connection</th><td>{escape(str(connection.get('type') or ''))} {escape(str(connection.get('base_url') or connection.get('folder') or ''))}</td></tr>
<tr><th>Preset</th><td>{escape(str(run.get('preset') or ''))}</td></tr>
<tr><th>Dataset</th><td>{escape(str(dataset.get('path') or ''))}</td></tr>
<tr><th>Dataset SHA-256</th><td>{escape(str(dataset.get('sha256') or ''))}</td></tr>
<tr><th>Rows</th><td>{escape(str(dataset.get('row_count') or ''))}</td></tr>
</table>
<h3>Suites</h3>
<table>
<tr><th>Suite</th><th>Source</th><th>Notes</th></tr>
{''.join(suite_rows)}
</table>

<h2>Results</h2>
<p>{escape(str(verdict))}</p>
<p>Failing live prompts: {escape(str(failure_count))}</p>
{fixture_note}
<p>Garak pass rate (1 - ASR): {escape(garak_rate_text)}. Attack success rate: {escape(garak_asr_text)}.</p>
<p>{escape(garak_wording)}</p>
<p>Garak report folder: {escape(str(run.get('garak_runs_dir') or ''))}</p>
<p>Run log: {escape(str(run.get('log_path') or ''))}</p>
<table>
<tr><th>Category</th><th>Status</th><th>Pass rate</th><th>Items</th></tr>
{''.join(category_rows)}
</table>

<h2>Findings</h2>
<p>{escape(narrative)}</p>
<p class="muted">Summary source: {escape(str(source_label))}. Number check: {escape(str(analysis.get('number_guard') or ''))}.</p>

<h2>Caveats</h2>
<ul>
<li>Rows marked fixture or smoke did not call the model. Do not treat them as a measurement of this model.</li>
<li>RAMPART is a smoke check in this version.</li>
<li>Retrieval is not run unless a retrieval set is added in a later version.</li>
<li>The number check keeps judge summaries from inventing figures. If it fails, the report uses the template summary.</li>
</ul>

<h2>Analysis</h2>
<p>{escape(narrative)}</p>

<h2>Failure appendix</h2>
<table>
<tr><th>Category</th><th>Source</th><th>Prompt</th><th>Response</th><th>Expected</th><th>Score</th><th>Matched span</th><th>Excerpt</th></tr>
{''.join(failure_rows)}
</table>
<h2>Evidence</h2>
<table>
<tr><th>Suite</th><th>Score</th><th>Match</th><th>Matched span</th><th>Excerpt</th><th>Detector note</th></tr>
{''.join(evidence_rows)}
</table>
</main>
</body>
</html>
"""
