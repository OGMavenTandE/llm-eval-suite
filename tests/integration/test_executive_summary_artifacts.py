from pathlib import Path
from unittest.mock import patch

from llm_eval.core.run_service import RunService
from llm_eval.evaluators.base import EvalResult
from llm_eval.reporting.reporter import EvalReporter


def _make_eval_result(metric_name: str, score: float, passed: bool) -> EvalResult:
    return EvalResult(metric_name=metric_name, score=score, passed=passed, details={})


def test_start_run_generates_executive_summary_artifact(valid_config: dict):
    output_dir = Path(valid_config["output_dir"])
    reporter = EvalReporter(output_dir=str(output_dir), run_name=valid_config["run_name"])
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
        result = service.start_run(valid_config)

    assert result.status == "completed"
    assert result.artifacts is not None
    assert result.artifacts.executive_summary_path is not None
    assert Path(result.artifacts.executive_summary_path).exists()
    assert Path(result.artifacts.report_manifest_path).exists()

    loaded = service.get_executive_summary(result.run_id, valid_config["output_dir"])
    assert loaded is not None
    assert loaded.run_id == result.run_id
    assert loaded.evaluation_purpose

    entry = service.get_run(result.run_id, valid_config["output_dir"])
    assert entry is not None
    assert entry["artifact_paths"]["executive_summary_path"] == result.artifacts.executive_summary_path
