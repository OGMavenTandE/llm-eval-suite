from datetime import datetime
from uuid import uuid4

from llm_eval.core.audit_service import AuditService
from llm_eval.core.config_service import ConfigService
from llm_eval.runner import EvalRunner
from llm_eval.schemas.run_request import RunValidationResult
from llm_eval.schemas.run_result import (
    ModelRunArtifacts,
    RunArtifactPaths,
    RunStartResult,
)
from llm_eval.storage.index import RunIndex


class RunService:
    """
    Main orchestration wrapper around EvalRunner.

    Exposes structured operations for CLI, future API, and UI layers.
    """

    def __init__(
        self,
        config_service: ConfigService | None = None,
        audit_service: AuditService | None = None,
    ):
        self.config_service = config_service or ConfigService()
        self.audit_service = audit_service or AuditService(self.config_service)

    def validate_run(self, config: dict) -> RunValidationResult:
        """Validate config and dataset without running inference."""
        return self.config_service.validate_run(config)

    def run_dry_run(self, config: dict, *, verbose: bool = False) -> RunStartResult:
        """Validate config via the runner dry-run path and return structured data."""
        started_at = datetime.now()
        run_id = uuid4().hex[:12]
        run_name = config.get("run_name", "eval_run")

        validation = self.validate_run(config)
        if not validation.valid:
            return RunStartResult(
                run_id=run_id,
                run_name=run_name,
                status="failed_validation",
                dry_run=True,
                started_at=started_at,
                completed_at=datetime.now(),
                validation=validation,
                message="; ".join(validation.errors),
            )

        runner = EvalRunner(config=config, dry_run=True, verbose=verbose)
        runner_result = runner.run()

        completed_at = datetime.now()
        message = validation.message
        if runner_result and runner_result.get("message"):
            message = runner_result["message"]

        result = RunStartResult(
            run_id=run_id,
            run_name=run_name,
            status="validated",
            dry_run=True,
            started_at=started_at,
            completed_at=completed_at,
            validation=validation,
            message=message,
        )

        output_dir = config.get("output_dir", "results/")
        RunIndex(output_dir).register_run(
            run_id=run_id,
            run_name=run_name,
            status="validated",
            dry_run=True,
            started_at=started_at,
            completed_at=completed_at,
            config_path=config.get("_config_path"),
        )

        return result

    def start_run(
        self,
        config: dict,
        *,
        verbose: bool = False,
        compare: bool = False,
    ) -> RunStartResult:
        """Execute a full evaluation run and return structured metadata."""
        started_at = datetime.now()
        run_id = uuid4().hex[:12]
        run_name = config.get("run_name", "eval_run")

        validation = self.validate_run(config)
        if not validation.valid:
            return RunStartResult(
                run_id=run_id,
                run_name=run_name,
                status="failed_validation",
                dry_run=False,
                started_at=started_at,
                completed_at=datetime.now(),
                validation=validation,
                message="; ".join(validation.errors),
            )

        runner = EvalRunner(config=config, dry_run=False, verbose=verbose, compare=compare)
        runner_result = runner.run() or {}

        artifacts = self._build_artifact_paths(config, runner_result)
        completed_at = datetime.now()

        result = RunStartResult(
            run_id=run_id,
            run_name=run_name,
            status="completed",
            dry_run=False,
            started_at=started_at,
            completed_at=completed_at,
            artifacts=artifacts,
            validation=validation,
            message=f"Run '{run_name}' completed.",
        )

        output_dir = config.get("output_dir", "results/")
        RunIndex(output_dir).register_run(
            run_id=run_id,
            run_name=run_name,
            status="completed",
            dry_run=False,
            started_at=started_at,
            completed_at=completed_at,
            config_path=config.get("_config_path"),
            artifact_paths=artifacts.model_dump() if artifacts else None,
        )

        return result

    def get_run_output_paths(self, run_result: RunStartResult) -> RunArtifactPaths | None:
        """Return artifact paths from a run result."""
        return run_result.artifacts

    @staticmethod
    def _build_artifact_paths(config: dict, runner_result: dict) -> RunArtifactPaths | None:
        model_artifacts = [
            ModelRunArtifacts(
                model_name=item["model_name"],
                run_dir=str(item["run_dir"]),
                summary_path=str(item["summary_path"]) if item.get("summary_path") else None,
                detailed_path=str(item["detailed_path"]) if item.get("detailed_path") else None,
            )
            for item in runner_result.get("model_runs", [])
        ]

        if not model_artifacts and not runner_result.get("comparison_dir"):
            return None

        comparison = runner_result.get("comparison", {}) or {}
        return RunArtifactPaths(
            run_name=config.get("run_name", "eval_run"),
            output_dir=config.get("output_dir", "results/"),
            model_artifacts=model_artifacts,
            comparison_dir=str(runner_result["comparison_dir"])
            if runner_result.get("comparison_dir")
            else None,
            comparison_summary_path=str(comparison.get("summary_path"))
            if comparison.get("summary_path")
            else None,
            comparison_detailed_path=str(comparison.get("detailed_path"))
            if comparison.get("detailed_path")
            else None,
        )
