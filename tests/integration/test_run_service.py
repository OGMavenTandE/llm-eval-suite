from pathlib import Path
from unittest.mock import patch

from llm_eval.core.run_service import RunService


def test_dry_run_persists_audit_metadata(valid_config: dict):
    service = RunService()
    result = service.run_dry_run(valid_config)

    assert result.audit is not None
    assert result.audit.audit_path is not None
    assert result.audit.status == "validated"
    assert result.audit.run_id == result.run_id
    assert result.audit.dataset_sample_count == 2
    assert result.audit.model_providers == ["ollama"]
    assert Path(result.audit.audit_path).exists()

    loaded = service.get_run_audit(result.run_id, valid_config["output_dir"])
    assert loaded is not None
    assert loaded.run_id == result.run_id


def test_list_and_get_run_after_dry_run(valid_config: dict):
    service = RunService()
    result = service.run_dry_run(valid_config)

    runs = service.list_runs(valid_config["output_dir"])
    assert len(runs) >= 1
    assert runs[0]["run_id"] == result.run_id

    entry = service.get_run(result.run_id, valid_config["output_dir"])
    assert entry is not None
    assert entry["audit_path"] is not None
    assert entry["status"] == "validated"
    assert entry["config_hash"] is not None


def test_start_run_handles_runtime_errors(valid_config: dict):
    service = RunService()

    with patch("llm_eval.core.run_service.EvalRunner") as mock_runner:
        mock_runner.return_value.run.side_effect = RuntimeError("model unavailable")
        result = service.start_run(valid_config)

    assert result.status == "failed_runtime"
    assert result.error_message == "model unavailable"
    assert result.output_dir == valid_config["output_dir"]
    assert result.started_at is not None
    assert result.completed_at is not None
    assert result.audit is not None
    assert result.audit.status == "failed_runtime"
    assert result.audit.error_message == "model unavailable"
    assert Path(result.audit.audit_path).exists()

    entry = service.get_run(result.run_id, valid_config["output_dir"])
    assert entry is not None
    assert entry["status"] == "failed_runtime"
    assert entry["error_message"] == "model unavailable"


def test_get_run_results_returns_none_for_missing_run(valid_config: dict):
    service = RunService()
    assert service.get_run_results("does-not-exist", valid_config["output_dir"]) is None


def test_get_run_artifacts_returns_none_without_artifacts(valid_config: dict):
    service = RunService()
    result = service.run_dry_run(valid_config)

    assert service.get_run_artifacts(result.run_id, valid_config["output_dir"]) is None
