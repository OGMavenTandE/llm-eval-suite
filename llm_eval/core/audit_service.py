import json
from datetime import datetime
from pathlib import Path

from llm_eval.core.config_service import ConfigService
from llm_eval.schemas.run_request import RunRequest, RunValidationResult
from llm_eval.schemas.run_result import AuditMetadata, RunStartResult


class AuditService:
    """Produce and persist structured audit metadata for evaluation runs."""

    AUDIT_DIRNAME = "audit"

    def __init__(self, config_service: ConfigService | None = None):
        self.config_service = config_service or ConfigService()

    def audit_dir(self, output_dir: str) -> Path:
        path = Path(output_dir) / self.AUDIT_DIRNAME
        path.mkdir(parents=True, exist_ok=True)
        return path

    def audit_file_path(self, output_dir: str, run_id: str) -> Path:
        return self.audit_dir(output_dir) / f"{run_id}.json"

    def build_audit_metadata(
        self,
        run_result: RunStartResult,
        config: dict,
        request: RunRequest | None = None,
    ) -> AuditMetadata:
        """Build an audit record from a completed, dry-run, or failed result."""
        request = request or self.config_service.normalize_config(
            config,
            dry_run=run_result.dry_run,
            compare=config.get("_compare", False),
            config_path=config.get("_config_path"),
        )
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
            error=run_result.error,
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
        error: str | None = None,
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
            error=error or ("; ".join(validation.errors) if validation.errors else None),
        )

    def persist(self, audit: AuditMetadata) -> AuditMetadata:
        """Write audit metadata to disk and return it with audit_path set."""
        path = self.audit_file_path(audit.output_dir, audit.run_id)
        payload = audit.model_dump(mode="json")
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)

        persisted = audit.model_copy(update={"audit_path": str(path)})
        with path.open("w", encoding="utf-8") as f:
            json.dump(persisted.model_dump(mode="json"), f, indent=2, default=str)
        return persisted

    def load(self, output_dir: str, run_id: str) -> AuditMetadata | None:
        """Load persisted audit metadata for a run."""
        path = self.audit_file_path(output_dir, run_id)
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return AuditMetadata.model_validate(data)
