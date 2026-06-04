import logging
import traceback
from datetime import datetime
from uuid import uuid4

from llm_eval.core.audit_service import AuditService
from llm_eval.core.config_service import ConfigService
from llm_eval.core.result_service import ResultService
from llm_eval.runner import EvalRunner
from llm_eval.schemas.run_result import (
    AuditMetadata,
    ModelRunArtifacts,
    RunArtifactPaths,
    RunStartResult,
)
from llm_eval.storage.index import RunIndex

logger = logging.getLogger(__name__)


class RunService:
    """
    Main orchestration wrapper around EvalRunner.

    Exposes structured operations for CLI, future API, and UI layers.
    """

    def __init__(
        self,
        config_service: ConfigService | None = None,
        audit_service: AuditService | None = None,
        result_service: ResultService | None = None,
    ):
        self.config_service = config_service or ConfigService()
        self.audit_service = audit_service or AuditService(self.config_service)
        self.result_service = result_service or ResultService()

    def validate_run(self, config: dict):
        """Validate config and dataset without running inference."""
        return self.config_service.validate_run(config)

    def run_dry_run(self, config: dict, *, verbose: bool = False) -> RunStartResult:
        """Validate config via the runner dry-run path and return structured data."""
        started_at = datetime.now()
        run_id = uuid4().hex[:12]
        run_name = config.get("run_name", "eval_run")
        output_dir = config.get("output_dir", "results/")

        validation = self.validate_run(config)
        if not validation.valid:
            result = RunStartResult(
                run_id=run_id,
                run_name=run_name,
                status="failed_validation",
                dry_run=True,
                started_at=started_at,
                completed_at=datetime.now(),
                validation=validation,
                message="; ".join(validation.errors),
                error="; ".join(validation.errors),
            )
            return self._finalize_run(result, config, output_dir)

        try:
            runner = EvalRunner(config=config, dry_run=True, verbose=verbose)
            runner_result = runner.run()
        except Exception as exc:
            logger.error("Dry-run failed: %s", exc)
            if verbose:
                traceback.print_exc()
            result = RunStartResult(
                run_id=run_id,
                run_name=run_name,
                status="failed",
                dry_run=True,
                started_at=started_at,
                completed_at=datetime.now(),
                validation=validation,
                message=f"Dry-run failed: {exc}",
                error=str(exc),
            )
            return self._finalize_run(result, config, output_dir)

        message = validation.message
        if runner_result and runner_result.get("message"):
            message = runner_result["message"]

        result = RunStartResult(
            run_id=run_id,
            run_name=run_name,
            status="validated",
            dry_run=True,
            started_at=started_at,
            completed_at=datetime.now(),
            validation=validation,
            message=message,
        )
        return self._finalize_run(result, config, output_dir)

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
        output_dir = config.get("output_dir", "results/")
        config["_compare"] = compare

        validation = self.validate_run(config)
        if not validation.valid:
            result = RunStartResult(
                run_id=run_id,
                run_name=run_name,
                status="failed_validation",
                dry_run=False,
                started_at=started_at,
                completed_at=datetime.now(),
                validation=validation,
                message="; ".join(validation.errors),
                error="; ".join(validation.errors),
            )
            return self._finalize_run(result, config, output_dir)

        try:
            runner = EvalRunner(config=config, dry_run=False, verbose=verbose, compare=compare)
            runner_result = runner.run() or {}
        except Exception as exc:
            logger.error("Run failed: %s", exc)
            if verbose:
                traceback.print_exc()
            result = RunStartResult(
                run_id=run_id,
                run_name=run_name,
                status="failed",
                dry_run=False,
                started_at=started_at,
                completed_at=datetime.now(),
                validation=validation,
                message=f"Run failed: {exc}",
                error=str(exc),
            )
            return self._finalize_run(result, config, output_dir)

        artifacts = self._build_artifact_paths(config, runner_result)
        result = RunStartResult(
            run_id=run_id,
            run_name=run_name,
            status="completed",
            dry_run=False,
            started_at=started_at,
            completed_at=datetime.now(),
            artifacts=artifacts,
            validation=validation,
            message=f"Run '{run_name}' completed.",
        )
        return self._finalize_run(result, config, output_dir)

    def list_runs(self, output_dir: str = "results/") -> list[dict]:
        """Return indexed runs for an output directory, newest first."""
        return RunIndex(output_dir).list_runs()

    def get_run(self, run_id: str, output_dir: str = "results/") -> dict | None:
        """Look up a single indexed run by ID."""
        return RunIndex(output_dir).get_run(run_id)

    def get_run_audit(self, run_id: str, output_dir: str = "results/") -> AuditMetadata | None:
        """Load persisted audit metadata for a run."""
        indexed = self.get_run(run_id, output_dir)
        if indexed and indexed.get("audit_path"):
            path = indexed["audit_path"]
            if path:
                try:
                    import json
                    from pathlib import Path

                    with Path(path).open(encoding="utf-8") as f:
                        data = json.load(f)
                    return AuditMetadata.model_validate(data)
                except (OSError, ValueError):
                    pass
        return self.audit_service.load(output_dir, run_id)

    def get_run_results(self, run_id: str, output_dir: str = "results/") -> dict | None:
        """
        Load result summaries and detailed traces for an indexed run.

        Returns None if the run is not found. Partial results are returned when
        only some artifact files exist.
        """
        entry = self.get_run(run_id, output_dir)
        if entry is None:
            return None

        artifact_paths = entry.get("artifact_paths") or {}
        model_results = []
        for model_entry in artifact_paths.get("model_artifacts", []):
            model_payload = {"model_name": model_entry.get("model_name"), "summary": None, "detailed": None}
            summary_path = model_entry.get("summary_path")
            detailed_path = model_entry.get("detailed_path")
            if summary_path:
                try:
                    model_payload["summary"] = self.result_service.load_summary(summary_path)
                except (FileNotFoundError, OSError, ValueError):
                    pass
            if detailed_path:
                try:
                    model_payload["detailed"] = self.result_service.load_detailed(detailed_path)
                except (FileNotFoundError, OSError, ValueError):
                    pass
            model_results.append(model_payload)

        comparison = None
        comparison_summary_path = artifact_paths.get("comparison_summary_path")
        comparison_detailed_path = artifact_paths.get("comparison_detailed_path")
        if comparison_summary_path or comparison_detailed_path:
            comparison = {"summary": None, "detailed": None}
            if comparison_summary_path:
                try:
                    comparison["summary"] = self.result_service.load_comparison_summary(
                        comparison_summary_path
                    )
                except (FileNotFoundError, OSError, ValueError):
                    pass
            if comparison_detailed_path:
                try:
                    comparison["detailed"] = self.result_service.load_comparison_detailed(
                        comparison_detailed_path
                    )
                except (FileNotFoundError, OSError, ValueError):
                    pass

        return {
            "run_id": run_id,
            "run_name": entry.get("run_name"),
            "status": entry.get("status"),
            "dry_run": entry.get("dry_run", False),
            "model_results": model_results,
            "comparison": comparison,
        }

    def get_run_output_paths(self, run_result: RunStartResult) -> RunArtifactPaths | None:
        """Return artifact paths from a run result."""
        return run_result.artifacts

    def _finalize_run(self, result: RunStartResult, config: dict, output_dir: str) -> RunStartResult:
        """Persist audit metadata and register the run in the index."""
        audit = self.audit_service.build_audit_metadata(result, config)
        audit = self.audit_service.persist(audit)

        RunIndex(output_dir).register_run(
            run_id=result.run_id,
            run_name=result.run_name,
            status=result.status,
            dry_run=result.dry_run,
            started_at=result.started_at,
            completed_at=result.completed_at,
            config_path=config.get("_config_path"),
            artifact_paths=result.artifacts.model_dump() if result.artifacts else None,
            audit_path=audit.audit_path,
            error=result.error,
        )

        return result.model_copy(update={"audit": audit})

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
