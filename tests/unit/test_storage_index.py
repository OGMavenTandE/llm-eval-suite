from datetime import datetime
from pathlib import Path

from llm_eval.core.audit_service import AuditService
from llm_eval.storage.artifacts import find_model_artifacts, standard_artifact_names
from llm_eval.storage.runs import RunDirectory, RunStorage


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
        dataset_sample_count=10,
        model_names=["m1"],
        model_providers=["ollama"],
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
    assert loaded.dataset_sample_count == 10
    assert loaded.model_providers == ["ollama"]
