"""One-command offline smoke. Uses a public fixture, not a model and not Dioptra."""

from datetime import datetime, timezone
from pathlib import Path

from llm_eval.dioptra.client import OfflineDioptraClient
from llm_eval.dioptra.export import ENTRYPOINT_NAME, write_bundle
from llm_eval.dioptra.schema import PilotBundle

# Illustrative numbers only. They are not measurements from a model.
FIXTURE_METRICS = (
    ("correctness_mean_score", 1.0),
    ("latency_mean_ms", 12.5),
)

DEFAULT_OUTPUT = Path("results") / "dioptra" / "smoke-experiment.json"


def run_smoke(output_path: str | Path | None = None) -> tuple[PilotBundle, Path]:
    """Record a one-job fixture experiment and write it as JSON."""
    client = OfflineDioptraClient(source="smoke-fixture")
    created_at = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
    client.experiments.create(
        name="llm-eval-dioptra-smoke",
        description=(
            "Public fixture for the Dioptra light pilot. "
            "Metric values are not model measurements."
        ),
        entrypoints=[ENTRYPOINT_NAME],
        created_at=created_at,
    )
    client.jobs.create(
        "llm-eval-dioptra-smoke",
        name="fixture-model",
        entrypoint_name=ENTRYPOINT_NAME,
        suite_status="validated",
        local_id="job-1",
        metrics_source="public-fixture",
        parameters={
            "dataset": "datasets/sample_correctness.jsonl",
            "evaluators": ["correctness", "latency"],
            "model": "fixture-model",
            "provider": "none",
            "dry_run": True,
        },
    )
    for metric_name, metric_value in FIXTURE_METRICS:
        client.jobs.append_metric("job-1", metric_name, metric_value, metric_step=0, timestamp=created_at)

    bundle = client.to_bundle()
    destination = Path(output_path) if output_path is not None else DEFAULT_OUTPUT
    written = write_bundle(bundle, destination)
    return bundle, written
