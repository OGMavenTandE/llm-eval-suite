from llm_eval.core.result_service import ResultService
from llm_eval.evaluators.base import EvalResult
from llm_eval.reporting.reporter import EvalReporter
from llm_eval.storage.artifacts import find_model_artifacts, standard_artifact_names


def _make_eval_result(metric_name: str, score: float, passed: bool) -> EvalResult:
    return EvalResult(
        metric_name=metric_name,
        score=score,
        passed=passed,
        details={},
    )


def test_reporter_creates_summary_and_detailed_outputs(tmp_path):
    output_dir = tmp_path / "results"
    reporter = EvalReporter(output_dir=str(output_dir), run_name="unit-test")

    reporter.record_result(
        sample_idx=0,
        prompt="What is 2+2?",
        expected="4",
        model_response_text="4",
        latency_ms=120.5,
        eval_results=[_make_eval_result("correctness", 1.0, True)],
    )

    detailed_path = reporter.save_detailed_results()
    summary_path = reporter.save_summary()

    assert summary_path.exists()
    assert detailed_path.exists()

    artifacts = find_model_artifacts(reporter.run_dir)
    assert artifacts["summary_path"] is not None
    assert artifacts["detailed_path"] is not None
    assert artifacts["summary_path"].name == standard_artifact_names()["summary"]
    assert artifacts["detailed_path"].name == standard_artifact_names()["detailed"]


def test_result_service_loads_reporter_outputs(tmp_path):
    output_dir = tmp_path / "results"
    reporter = EvalReporter(output_dir=str(output_dir), run_name="load-test")

    reporter.record_result(
        sample_idx=0,
        prompt="Capital of France?",
        expected="Paris",
        model_response_text="Paris",
        latency_ms=95.0,
        eval_results=[
            _make_eval_result("correctness", 1.0, True),
            _make_eval_result("latency", 1.0, True),
        ],
    )

    summary_path = reporter.save_summary()
    detailed_path = reporter.save_detailed_results()

    result_service = ResultService()
    summary_rows = result_service.load_summary(summary_path)
    detailed_rows = result_service.load_detailed(detailed_path)

    assert len(summary_rows) == 2
    metric_names = {row["metric_name"] for row in summary_rows}
    assert metric_names == {"correctness", "latency"}

    assert len(detailed_rows) == 1
    assert detailed_rows[0]["prompt"] == "Capital of France?"
    assert len(detailed_rows[0]["evaluations"]) == 2

    resolved = result_service.resolve_model_artifacts(reporter.run_dir)
    assert resolved["summary_path"] == summary_path
    assert resolved["detailed_path"] == detailed_path
