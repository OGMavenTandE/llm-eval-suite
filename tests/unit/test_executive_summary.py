from llm_eval.evaluators.base import EvalResult
from llm_eval.reporting.executive_summary import build_executive_summary
from llm_eval.reporting.reporter import EvalReporter


def _make_eval_result(metric_name: str, score: float, passed: bool) -> EvalResult:
    return EvalResult(metric_name=metric_name, score=score, passed=passed, details={})


def test_build_executive_summary_from_model_results():
    model_results = [
        {
            "model_name": "test-model",
            "summary": [
                {
                    "metric_name": "correctness",
                    "mean_score": "0.50",
                    "pass_rate": "50%",
                    "sample_count": "2",
                }
            ],
            "detailed": [
                {
                    "sample_idx": 0,
                    "prompt": "What is 2+2?",
                    "expected": "4",
                    "model_response": "4",
                    "evaluations": [{"metric_name": "correctness", "score": 1.0, "passed": True}],
                },
                {
                    "sample_idx": 1,
                    "prompt": "Capital of France?",
                    "expected": "Paris",
                    "model_response": "Lyon",
                    "evaluations": [{"metric_name": "correctness", "score": 0.0, "passed": False}],
                },
            ],
        }
    ]

    summary = build_executive_summary(
        run_id="run123",
        run_name="unit-test",
        dataset_path="/datasets/sample.jsonl",
        model_names=["test-model"],
        evaluator_names=["correctness"],
        model_results=model_results,
    )

    assert summary.run_id == "run123"
    assert "test-model" in summary.evaluation_purpose
    assert summary.needs_human_review_count == 1
    assert summary.key_strengths
    assert summary.key_weaknesses
    assert summary.notable_metrics[0].label == "Answer accuracy"


def test_report_artifact_service_writes_executive_summary(tmp_path):
    output_dir = tmp_path / "results"
    reporter = EvalReporter(output_dir=str(output_dir), run_name="artifact-test")
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

    from llm_eval.reporting.report_artifacts import ReportArtifactService, load_executive_summary, load_report_manifest
    from llm_eval.schemas.run_result import ModelRunArtifacts, RunArtifactPaths

    artifacts = RunArtifactPaths(
        run_name="artifact-test",
        output_dir=str(output_dir),
        model_artifacts=[
            ModelRunArtifacts(
                model_name="test-model",
                run_dir=str(reporter.run_dir),
                summary_path=str(summary_path),
                detailed_path=str(detailed_path),
            )
        ],
    )

    service = ReportArtifactService()
    updated = service.generate_and_persist(
        run_id="run456",
        run_name="artifact-test",
        dataset_path="/datasets/sample.jsonl",
        model_names=["test-model"],
        evaluator_names=["correctness"],
        artifacts=artifacts,
    )

    assert updated.executive_summary_path is not None
    assert updated.report_manifest_path is not None
    assert updated.reports_dir is not None

    loaded = load_executive_summary(updated.executive_summary_path)
    assert loaded.run_id == "run456"
    assert loaded.overall_outcome

    manifest = load_report_manifest(updated.report_manifest_path)
    kinds = {entry.kind for entry in manifest.artifacts}
    assert "executive_summary" in kinds
    assert "summary" in kinds
