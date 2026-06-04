from llm_eval.core.run_service import RunService


def test_dry_run_succeeds_on_valid_config(valid_config: dict):
    service = RunService()
    result = service.run_dry_run(valid_config)

    assert result.dry_run is True
    assert result.status == "validated"
    assert result.validation is not None
    assert result.validation.valid is True
    assert result.validation.dataset is not None
    assert result.validation.dataset.sample_count == 2
    assert len(result.validation.models) == 1
    assert result.validation.evaluator_names == ["correctness", "latency"]
    assert result.message is not None
    assert "Config valid" in result.message or "[dry-run]" in result.message


def test_validate_run_returns_structured_errors(valid_config: dict):
    service = RunService()
    bad_config = dict(valid_config)
    bad_config["dataset"] = "does-not-exist.jsonl"

    result = service.validate_run(bad_config)
    assert result.valid is False
    assert result.errors
    assert result.dataset is None


def test_dry_run_fails_on_invalid_config(valid_config: dict):
    service = RunService()
    bad_config = dict(valid_config)
    bad_config["models"] = []

    result = service.run_dry_run(bad_config)
    assert result.status == "failed_validation"
    assert result.validation is not None
    assert result.validation.valid is False
