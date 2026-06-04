import json
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from llm_eval.core.audit_service import AuditService
from llm_eval.storage.artifacts import find_model_artifacts, standard_artifact_names
from llm_eval.storage.index import RunIndex
from llm_eval.storage.runs import RunDirectory, RunStorage


def test_run_index_register_and_list(tmp_path: Path):
    output_dir = tmp_path / "results"
    index = RunIndex(str(output_dir))

    first = index.register_run(
        run_id="abc123",
        run_name="test-run",
        status="validated",
        dry_run=True,
        started_at=datetime(2026, 1, 1, 12, 0, 0),
        completed_at=datetime(2026, 1, 1, 12, 0, 1),
        config_path="config/example.yaml",
        audit_path=str(output_dir / "audit" / "abc123.json"),
    )
    second = index.register_run(
        run_id="def456",
        run_name="later-run",
        status="completed",
        dry_run=False,
        started_at=datetime(2026, 1, 2, 12, 0, 0),
    )

    assert first["run_id"] == "abc123"
    assert second["run_id"] == "def456"

    runs = index.list_runs()
    assert len(runs) == 2
    assert runs[0]["run_id"] == "def456"
    assert runs[1]["run_id"] == "abc123"

    assert index.index_path.exists()
    with index.index_path.open(encoding="utf-8") as f:
        data = json.load(f)
    assert isinstance(data, list)
    assert len(data) == 2


def test_run_index_get_and_update(tmp_path: Path):
    output_dir = tmp_path / "results"
    index = RunIndex(str(output_dir))

    index.register_run(
        run_id="run-1",
        run_name="alpha",
        status="validated",
        dry_run=True,
    )
    index.register_run(
        run_id="run-1",
        run_name="alpha",
        status="completed",
        dry_run=False,
        error=None,
    )

    entry = index.get_run("run-1")
    assert entry is not None
    assert entry["status"] == "completed"
    assert index.get_run("missing") is None

    runs = index.list_runs()
    assert len(runs) == 1


def test_run_storage_create_directory(tmp_path: Path):
    storage = RunStorage(str(tmp_path / "results"))
    run_dir = storage.create_run_directory("my-run", run_id="fixed-id")

    assert run_dir.run_id == "fixed-id"
    assert run_dir.base_dir.exists()
    assert run_dir.base_dir.name.startswith("my-run_")
    assert run_dir.metadata_path.parent == run_dir.base_dir


def test_standard_artifact_names():
    names = standard_artifact_names()
    assert names["summary"] == RunDirectory.SUMMARY_FILENAME
    assert names["detailed"] == RunDirectory.DETAILED_FILENAME
    assert names["metadata"] == RunDirectory.METADATA_FILENAME


def test_find_model_artifacts_missing_files(tmp_path: Path):
    run_dir = tmp_path / "empty-run"
    run_dir.mkdir()

    artifacts = find_model_artifacts(run_dir)
    assert artifacts["summary_path"] is None
    assert artifacts["detailed_path"] is None


def test_audit_service_persist_and_load(tmp_path: Path):
    from llm_eval.schemas.run_result import AuditMetadata

    output_dir = str(tmp_path / "results")
    audit = AuditMetadata(
        run_id="audit-1",
        run_name="test",
        started_at=datetime(2026, 1, 1, 12, 0, 0),
        status="validated",
        output_dir=output_dir,
        config_hash="abc123",
        dataset_path="datasets/sample.jsonl",
        model_names=["m1"],
        evaluator_names=["correctness"],
    )

    service = AuditService()
    persisted = service.persist(audit)

    assert persisted.audit_path is not None
    assert Path(persisted.audit_path).exists()

    loaded = service.load(output_dir, "audit-1")
    assert loaded is not None
    assert loaded.run_id == "audit-1"
    assert loaded.config_hash == "abc123"
    assert loaded.audit_path is not None
