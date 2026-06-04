from datetime import datetime
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from llm_eval.schemas.run_request import RunValidationResult


class ModelRunArtifacts(BaseModel):
    """Artifact paths for a single model within a run."""

    model_name: str
    run_dir: str
    summary_path: str | None = None
    detailed_path: str | None = None


class RunArtifactPaths(BaseModel):
    """Filesystem locations for run outputs."""

    run_name: str
    output_dir: str
    model_artifacts: list[ModelRunArtifacts] = Field(default_factory=list)
    comparison_dir: str | None = None
    comparison_summary_path: str | None = None
    comparison_detailed_path: str | None = None


class RunStartResult(BaseModel):
    """Structured result after starting or completing a run."""

    run_id: str
    run_name: str
    status: str
    dry_run: bool = False
    output_dir: str = "results/"
    started_at: datetime
    completed_at: datetime | None = None
    artifacts: RunArtifactPaths | None = None
    validation: "RunValidationResult | None" = None
    audit: "AuditMetadata | None" = None
    message: str | None = None
    error_message: str | None = None


class AuditMetadata(BaseModel):
    """Structured audit record for a run."""

    run_id: str
    run_name: str
    started_at: datetime
    completed_at: datetime | None = None
    config_path: str | None = None
    config_hash: str | None = None
    dataset_path: str | None = None
    dataset_sample_count: int | None = None
    model_names: list[str] = Field(default_factory=list)
    model_providers: list[str] = Field(default_factory=list)
    evaluator_names: list[str] = Field(default_factory=list)
    dry_run: bool = False
    compare: bool = False
    output_dir: str = "results/"
    status: str = "unknown"
    artifact_paths: RunArtifactPaths | None = None
    audit_path: str | None = None
    error_message: str | None = None


from llm_eval.schemas.run_request import RunValidationResult  # noqa: E402

RunStartResult.model_rebuild(
    _types_namespace={
        "RunValidationResult": RunValidationResult,
        "AuditMetadata": AuditMetadata,
    }
)
