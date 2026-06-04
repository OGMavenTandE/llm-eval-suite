from pathlib import Path
from unittest.mock import patch

from llm_eval.core.run_service import RunService
from llm_eval.evaluators.base import EvalResult
from llm_eval.reporting.reporter import EvalReporter


def _make_eval_result(metric_name: str, score: float, passed: bool) -> EvalResult:
    return EvalResult(metric_name=metric_name, score=score, passed=passed, details={})


def _create_completed_run(api_workspace, valid_config):
    output_dir = api_workspace.output_dir
    reporter = EvalReporter(output_dir=str(output_dir), run_name="api-report-test")
    reporter.record_result(
        sample_idx=0,
        prompt="What is 2+2?",
        expected="4",
        model_response_text="4",
        latency_ms=100.0,
        eval_results=[_make_eval_result("correctness", 1.0, True)],
    )
    summary_path = reporter.save_summary()
    detailed_path = reporter.save_detailed_results()

    config = {
        **valid_config,
        "output_dir": str(output_dir),
        "run_name": "api-report-test",
    }
    runner_result = {
        "model_runs": [
            {
                "model_name": "test-model",
                "run_dir": str(reporter.run_dir),
                "summary_path": str(summary_path),
                "detailed_path": str(detailed_path),
            }
        ],
        "comparison_dir": None,
        "comparison": None,
    }

    service = RunService()
    with patch("llm_eval.core.run_service.EvalRunner") as mock_runner:
        mock_runner.return_value.run.return_value = runner_result
        return service.start_run(config)


def test_executive_summary_endpoint_returns_summary(client, api_workspace, valid_config):
    result = _create_completed_run(api_workspace, valid_config)

    response = client.get(f"/runs/{result.run_id}/reports/executive-summary")
    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == result.run_id
    assert payload["ready"] is True
    assert payload["summary"]["run_id"] == result.run_id
    assert payload["summary"]["overall_outcome"]
    assert payload["summary"]["recommended_next_step"]


def test_executive_summary_not_ready_for_dry_run(client, api_workspace):
    dataset_path = api_workspace.datasets_dir / "sample.jsonl"
    create_response = client.post(
        "/runs",
        json={
            "dry_run": True,
            "output_dir": str(api_workspace.output_dir),
            "config": {
                "dataset": str(dataset_path),
                "models": [{"name": "test-model", "provider": "ollama"}],
                "evaluators": [{"name": "correctness", "mode": "exact_match", "threshold": 0.8}],
            },
        },
    )
    run_id = create_response.json()["run_id"]

    from tests.api.conftest import wait_for_terminal_status

    wait_for_terminal_status(client, run_id)

    response = client.get(f"/runs/{run_id}/reports/executive-summary")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ready"] is False
    assert payload["summary"] is None


def test_executive_summary_missing_run_returns_404(client):
    response = client.get("/runs/does-not-exist/reports/executive-summary")
    assert response.status_code == 404


def test_artifacts_include_executive_summary_file(client, api_workspace, valid_config):
    result = _create_completed_run(api_workspace, valid_config)

    response = client.get(f"/runs/{result.run_id}/artifacts")
    assert response.status_code == 200
    kinds = {item["kind"] for item in response.json()["files"]}
    assert "executive_summary" in kinds
    assert "report_manifest" in kinds
