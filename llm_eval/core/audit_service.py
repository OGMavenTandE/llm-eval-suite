from datetime import datetime

from llm_eval.core.config_service import ConfigService
from llm_eval.schemas.run_request import RunRequest, RunValidationResult
from llm_eval.schemas.run_result import AuditMetadata, RunArtifactPaths, RunStartResult


class AuditService:
    """Produce structured audit metadata for evaluation runs."""

    def __init__(self, config_service: ConfigService | None = None):
        self.config_service = config_service or ConfigService()

    def build_audit_metadata(
        self,
        run_result: RunStartResult,
        config: dict,
        request: RunRequest | None = None,
    ) -> AuditMetadata:
        """Build an audit record from a completed or dry-run result."""
        request = request or self.config_service.normalize_config(config)
        model_names = [m.name for m in request.models]
        evaluator_names = [ev.get("name", "") for ev in request.evaluators if ev.get("name")]

        return AuditMetadata(
            run_id=run_result.run_id,
            run_name=run_result.run_name,
            started_at=run_result.started_at,
            completed_at=run_result.completed_at,
            config_path=request.config_path,
            config_hash=self.config_service.config_hash(config),
            dataset_path=request.dataset,
            model_names=model_names,
            evaluator_names=evaluator_names,
            dry_run=run_result.dry_run,
            compare=request.compare,
            output_dir=request.output_dir,
            status=run_result.status,
            artifact_paths=run_result.artifacts,
        )

    def build_from_validation(
        self,
        run_id: str,
        run_name: str,
        validation: RunValidationResult,
        config: dict,
        *,
        config_path: str | None = None,
        started_at: datetime | None = None,
    ) -> AuditMetadata:
        """Build audit metadata for a validation-only or dry-run outcome."""
        started = started_at or datetime.now()
        request = self.config_service.normalize_config(config, config_path=config_path)

        return AuditMetadata(
            run_id=run_id,
            run_name=run_name,
            started_at=started,
            completed_at=datetime.now(),
            config_path=config_path,
            config_hash=self.config_service.config_hash(config),
            dataset_path=request.dataset,
            model_names=[m.name for m in validation.models],
            evaluator_names=validation.evaluator_names,
            dry_run=True,
            compare=request.compare,
            output_dir=request.output_dir,
            status="validated" if validation.valid else "failed_validation",
            artifact_paths=None,
        )
