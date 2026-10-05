import json
import math
from datetime import datetime, timezone

import pytest

from llm_eval.dioptra.cli import main
from llm_eval.dioptra.client import OfflineDioptraClient, normalize_metric_value
from llm_eval.dioptra.export import export_audit_file, metrics_from_summary_csv
from llm_eval.dioptra.schema import PilotBundle
from llm_eval.dioptra.smoke import FIXTURE_METRICS, run_smoke
from llm_eval.schemas.run_result import AuditMetadata, ModelRunArtifacts, RunArtifactPaths


def test_smoke_writes_offline_fixture(tmp_path):
    output = tmp_path / "smoke-experiment.json"
    bundle, written = run_smoke(output)

    assert written == output
    assert bundle.live_api is False
    assert bundle.source == "smoke-fixture"
    assert bundle.experiment.group_id is None
    assert bundle.experiment.jobs[0].submitted is False
    assert bundle.experiment.jobs[0].metrics_source == "public-fixture"
    assert bundle.experiment.jobs[0].dioptra_status_if_submitted is None
    recorded = [(item.name, item.value) for item in bundle.experiment.jobs[0].metrics]
    assert recorded == list(FIXTURE_METRICS)

    loaded = PilotBundle.model_validate(json.loads(output.read_text(encoding="utf-8")))
    assert loaded.schema_version == "dioptra-shaped-0.1"
    metric_posts = [item for item in loaded.rest_examples if item.path.endswith("/metrics")]
    assert metric_posts[0].body == {
        "name": "correctness_mean_score",
        "value": 1.0,
        "step": 0,
        "timestamp": "2026-01-15T12:00:00+00:00",
    }
    assert "group_id" in loaded.rest_examples[0].body
    assert loaded.rest_examples[0].body["group_id"] is None
    assert loaded.rest_examples[0].blocked_reason


def test_smoke_cli(tmp_path, capsys):
    output = tmp_path / "nested" / "smoke.json"
    main(["smoke", "--output", str(output)])
    captured = capsys.readouterr()
    assert "live_api=false" in captured.out
    assert output.is_file()


def test_metric_value_encoding():
    assert normalize_metric_value(1.5) == 1.5
    assert normalize_metric_value(float("nan")) == "nan"
    assert normalize_metric_value(float("inf")) == "inf"
    assert normalize_metric_value(float("-inf")) == "-inf"
    assert normalize_metric_value("nan") == "nan"


def test_client_rejects_duplicate_metric_and_live_flags():
    client = OfflineDioptraClient()
    client.experiments.create("exp", entrypoints=["yaml-eval"])
    client.jobs.create(
        "exp",
        name="model",
        entrypoint_name="yaml-eval",
        suite_status="completed",
        metrics_source="none",
    )
    client.jobs.append_metric("job-1", "loss", math.nan)
    with pytest.raises(ValueError, match="primary key"):
        client.jobs.append_metric("job-1", "loss", 0.1, metric_step=0)

    with pytest.raises(ValueError, match="already exists"):
        client.experiments.create("exp")

    with pytest.raises(ValueError):
        PilotBundle.model_validate(
            {
                "source": "x",
                "live_api": True,
                "experiment": {
                    "name": "exp",
                    "created_at": "2026-01-01T00:00:00Z",
                },
            }
        )


def test_export_audit_maps_summary_csv(tmp_path):
    summary = tmp_path / "results_summary.csv"
    summary.write_text(
        "metric_name,mean_score,pass_rate,sample_count\n"
        "correctness,0.8,1.0,2\n",
        encoding="utf-8",
    )
    assert metrics_from_summary_csv(summary) == [
        ("correctness_mean_score", 0.8),
        ("correctness_pass_rate", 1.0),
    ]

    audit = AuditMetadata(
        run_id="run-9",
        run_name="pilot-export",
        started_at=datetime(2026, 2, 1, tzinfo=timezone.utc),
        status="completed",
        output_dir=str(tmp_path),
        config_hash="abc",
        config_path="config/example_eval.yaml",
        dataset_path="datasets/sample_correctness.jsonl",
        dataset_sample_count=10,
        model_names=["fixture-model"],
        model_providers=["ollama"],
        evaluator_names=["correctness"],
        artifact_paths=RunArtifactPaths(
            run_name="pilot-export",
            output_dir=str(tmp_path),
            model_artifacts=[
                ModelRunArtifacts(
                    model_name="fixture-model",
                    run_dir=str(tmp_path),
                    summary_path=str(summary),
                )
            ],
        ),
    )
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit.model_dump(mode="json")), encoding="utf-8")

    bundle = export_audit_file(audit_path)
    job = bundle.experiment.jobs[0]
    assert bundle.source == "audit"
    assert job.submitted is False
    assert job.dioptra_status_if_submitted == "finished"
    assert job.metrics_source == "results_summary.csv"
    assert [(item.name, item.value) for item in job.metrics] == [
        ("correctness_mean_score", 0.8),
        ("correctness_pass_rate", 1.0),
    ]
    assert job.parameters["config_hash"] == "abc"
    assert job.parameters["provider"] == "ollama"


def test_export_audit_warns_when_summary_missing(tmp_path, capsys):
    audit = AuditMetadata(
        run_id="run-10",
        run_name="dry",
        started_at=datetime(2026, 2, 2, tzinfo=timezone.utc),
        status="validated",
        output_dir=str(tmp_path),
        dry_run=True,
        model_names=["m1"],
        artifact_paths=RunArtifactPaths(
            run_name="dry",
            output_dir=str(tmp_path),
            model_artifacts=[
                ModelRunArtifacts(
                    model_name="m1",
                    run_dir=str(tmp_path),
                    summary_path=str(tmp_path / "missing.csv"),
                )
            ],
        ),
    )
    audit_path = tmp_path / "audit.json"
    audit_path.write_text(json.dumps(audit.model_dump(mode="json")), encoding="utf-8")
    output = tmp_path / "out.json"
    main(["export-audit", str(audit_path), "--output", str(output)])
    captured = capsys.readouterr()
    assert "Summary path missing" in captured.err

    bundle = PilotBundle.model_validate(json.loads(output.read_text(encoding="utf-8")))
    assert bundle.experiment.jobs[0].metrics == []
    assert bundle.experiment.jobs[0].metrics_source == "none"
    assert bundle.experiment.jobs[0].dioptra_status_if_submitted is None
    assert bundle.warnings


def test_export_audit_cli_missing_file(tmp_path):
    with pytest.raises(SystemExit) as exc:
        main(["export-audit", str(tmp_path / "nope.json"), "--output", str(tmp_path / "out.json")])
    assert exc.value.code == 1
