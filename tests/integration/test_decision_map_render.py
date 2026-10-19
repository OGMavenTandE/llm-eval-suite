"""Example map through the CLI, and the one-page HTML export."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from llm_eval.decision_map.cli import main
from llm_eval.decision_map.load import load_map, load_runs
from llm_eval.decision_map.render import render_html
from llm_eval.decision_map.resolve import resolve_map
from llm_eval.decision_map.schema import Attestation, Decision, DecisionMap, EvidenceRequirement, SystemInfo
from llm_eval.decision_map.status import as_utc

EXAMPLE = Path(__file__).resolve().parents[2] / "llm_eval" / "decision_map" / "examples" / "wildfire_edge_assist.yaml"
RUN = EXAMPLE.parent / "wildfire_run"
AS_OF = "2026-10-19"


def test_example_cli_renders_one_page_without_granting_authority(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    html_path = tmp_path / "wildfire.html"
    json_path = tmp_path / "wildfire.json"
    main(["validate", str(EXAMPLE)])
    main(["resolve", str(EXAMPLE), "--runs", str(RUN), "--as-of", AS_OF, "--output", str(json_path)])
    main(["render", str(EXAMPLE), "--runs", str(RUN), "--as-of", AS_OF, "--output", str(html_path)])
    captured = capsys.readouterr()
    html = html_path.read_text(encoding="utf-8")
    payload = json_path.read_text(encoding="utf-8")
    combined = "\n".join([html, payload, captured.out, captured.err])
    assert "authority granted" not in combined.casefold()

    statuses = {row["id"]: row["status"] for row in json.loads(payload)["decisions"]}
    assert statuses == {
        "R1": "partial",
        "R2": "met",
        "R3": "met",
        "R4": "partial",
        "R5": "gap",
        "R6": "not_assessed",
    }
    assert "Keep" not in payload
    assert "Advisory only" in payload
    assert "Human on the loop" in payload
    assert "Not authorized" in payload

    assert 'class="report-doc"' in html
    assert "data:font/woff2;base64" in html
    assert "Public Sans" in html
    assert "@page { size: 11in 8.5in; margin: 0; }" in html
    assert "Decision-to-Evidence Map" in html
    assert "Wildfire Edge Assist" in html
    assert "EXAMPLE" in html
    assert "Recommended authority" in html
    assert "(decision authority grants)" in html
    assert "example-wildfire" in html
    assert "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee" in html
    assert "95% lower bound" in html
    assert html.count("<section") == 1
    assert "sheet dense" not in html
    for marker in ("st par", "st met", "st gap", "st na"):
        assert marker in html
    assert "Print to PDF from the browser" in captured.out
    assert "OK 6 decisions" in captured.out


def test_overflow_paginates_after_seven_rows():
    seven = _manual_map(7)
    eight = _manual_map(8)
    now = as_utc(date(2026, 10, 19))
    seven_html = render_html(seven, resolve_map(seven, [], now=now))
    eight_html = render_html(eight, resolve_map(eight, [], now=now))
    assert seven_html.count("<section") == 1
    assert "sheet dense" in seven_html
    assert eight_html.count("<section") == 2
    assert eight_html.count("<h1>Decision-to-Evidence Map</h1>") == 2
    assert "authority granted" not in seven_html.casefold()
    assert "authority granted" not in eight_html.casefold()


def test_validate_failure_exits(tmp_path: Path):
    path = tmp_path / "bad.yaml"
    path.write_text("schema_version: 1\n", encoding="utf-8")
    with pytest.raises(SystemExit) as caught:
        main(["validate", str(path)])
    assert caught.value.code == 1


def _manual_map(count: int) -> DecisionMap:
    decisions = []
    for index in range(count):
        decisions.append(
            Decision(
                id=f"R{index + 1}",
                decision=f"Decision {index + 1}",
                risk="low",
                recommended_authority="Advisory only",
                evidence=[
                    EvidenceRequirement(
                        id="e",
                        label="Signed note",
                        manual=True,
                        attestation=Attestation(
                            who="Example attestant",
                            date=date(2026, 10, 1),
                            signature="EXAMPLE",
                            status="met",
                        ),
                    )
                ],
            )
        )
    return DecisionMap(schema_version=1, system=SystemInfo(name="Demo"), decisions=decisions)


def test_example_files_are_what_the_cli_reads():
    decision_map = load_map(EXAMPLE)
    runs = load_runs(RUN)
    assert decision_map.illustrative is True
    assert decision_map.system.name == "Wildfire Edge Assist"
    assert runs[0].run_id == "example-wildfire"
    assert runs[0].items is None
