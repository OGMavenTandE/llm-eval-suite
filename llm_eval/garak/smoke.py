"""One-command offline smoke. Scores the canned probe file, not a model."""

import json
from datetime import datetime, timezone
from pathlib import Path
import uuid

from llm_eval.garak.schema import (
    CompletionEntry,
    InitEntry,
    SetupEntry,
    SmokeReport,
)
from llm_eval.garak.score import FixtureRow, score_fixture

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FIXTURE = REPO_ROOT / "fixtures" / "garak" / "probe-responses.jsonl"
DEFAULT_OUTPUT = Path("results") / "garak" / "smoke-report.jsonl"


def load_fixture(path: str | Path) -> list[FixtureRow]:
    fixture_path = Path(path)
    if not fixture_path.is_file():
        raise FileNotFoundError(f"garak fixture not found: {fixture_path}")
    rows: list[FixtureRow] = []
    for line_number, line in enumerate(
        fixture_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{fixture_path}:{line_number}: {exc}") from exc
        try:
            rows.append(FixtureRow.model_validate(payload))
        except ValueError as exc:
            raise ValueError(f"{fixture_path}:{line_number}: {exc}") from exc
    return rows


def build_report(rows: list[FixtureRow], *, run_id: str | None = None) -> SmokeReport:
    """Score ``rows`` and wrap them in a garak-shaped report. No I/O."""
    if run_id is None:
        run_id = str(uuid.uuid4())
    started = datetime.now(timezone.utc).isoformat()
    attempts, evals, summaries = score_fixture(rows)
    probe_spec = ",".join(sorted({row.probe_classname for row in rows}))
    return SmokeReport(
        setup=SetupEntry(
            source="smoke-fixture",
            probe_spec=probe_spec,
            run_id=run_id,
            start_time=started,
        ),
        init=InitEntry(start_time=started, run=run_id),
        attempts=attempts,
        evals=evals,
        summaries=summaries,
        completion=CompletionEntry(
            end_time=datetime.now(timezone.utc).isoformat(),
            run=run_id,
        ),
    )


def write_report(report: SmokeReport, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.to_jsonl(), encoding="utf-8")
    return path


def run_smoke(
    output_path: str | Path | None = None,
    fixture_path: str | Path | None = None,
) -> tuple[SmokeReport, Path]:
    """Score the canned probe/response file and write ``results/garak`` JSONL."""
    rows = load_fixture(fixture_path if fixture_path is not None else DEFAULT_FIXTURE)
    report = build_report(rows)
    destination = Path(output_path) if output_path is not None else DEFAULT_OUTPUT
    written = write_report(report, destination)
    return report, written
