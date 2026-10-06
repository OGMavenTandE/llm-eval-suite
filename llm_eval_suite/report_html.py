"""Standalone HTML report. Print to PDF from the browser. Styling is the token file."""

from __future__ import annotations

from html import escape
from pathlib import Path

TOKENS_PATH = Path(__file__).parent / "static" / "tokens.css"


def render_report(run: dict, items: list[dict]) -> str:
    css = ""
    if TOKENS_PATH.is_file():
        css = TOKENS_PATH.read_text(encoding="utf-8")
    connection = run.get("connection") or {}
    dataset = run.get("dataset") or {}
    card = run.get("scorecard") or {}
    analysis = run.get("analysis") or {}
    suites = run.get("suites") or []
    failures = [item for item in items if not item.get("passed")]

    category_rows = []
    for row in card.get("categories") or []:
        rate = "not run" if row.get("pass_rate") is None else f"{row['pass_percent']}%"
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
            "</tr>"
        )
    if not failure_rows:
        failure_rows.append("<tr><td colspan='6'>No failing prompts were stored.</td></tr>")

    overall = card.get("overall_pass_percent")
    overall_text = "not available" if overall is None else f"{overall}%"
    narrative = analysis.get("narrative") or "No analysis has been written yet."
    source_label = analysis.get("source_label") or "not run"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
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
<p>Overall pass rate (live categories): {escape(overall_text)}</p>
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
<tr><th>Category</th><th>Source</th><th>Prompt</th><th>Response</th><th>Expected</th><th>Score</th></tr>
{''.join(failure_rows)}
</table>
</main>
</body>
</html>
"""
