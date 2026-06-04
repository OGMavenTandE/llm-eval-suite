from unittest.mock import patch

import pytest

from llm_eval.core.run_service import RunService


def test_dry_run_persists_audit_metadata(valid_config: dict):
    service = RunService()
    result = service.run_dry_run(valid_config)

    assert result.audit is not None
    assert result.audit.audit_path is not None
    assert result.audit.status == "validated"
    assert result.audit.run_id == result.run_id

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


def test_start_run_handles_runtime_errors(valid_config: dict):
    service = RunService()

    with patch("llm_eval.core.run_service.EvalRunner") as mock_runner:
        mock_runner.return_value.run.side_effect = RuntimeError("model unavailable")
        result = service.start_run(valid_config)

    assert result.status == "failed"
    assert result.error == "model unavailable"
    assert result.audit is not None
    assert result.audit.status == "failed"
    assert result.audit.error == "model unavailable"

    entry = service.get_run(result.run_id, valid_config["output_dir"])
    assert entry is not None
    assert entry["status"] == "failed"
    assert entry["error"] == "model unavailable"


def test_get_run_results_returns_none_for_missing_run(valid_config: dict):
    service = RunService()
    assert service.get_run_results("does-not-exist", valid_config["output_dir"]) is None
