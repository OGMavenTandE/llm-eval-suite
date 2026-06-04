from datetime import datetime

from pydantic import BaseModel, Field

from apps.api.schemas.common import LinkRef


class CreateRunRequest(BaseModel):
    config_path: str | None = None
    config: dict | None = None
    dry_run: bool = False
    run_name: str | None = None
    compare: bool = False
    output_dir: str | None = None


class CreateRunResponse(BaseModel):
    run_id: str
    status: str
    message: str | None = None
    created_at: datetime
    output_dir: str
    dry_run: bool = False
    links: list[LinkRef] = Field(default_factory=list)


class RunSummaryResponse(BaseModel):
    run_id: str
    run_name: str
    status: str
    dry_run: bool = False
    output_dir: str | None = None
    started_at: str | None = None
    completed_at: str | None = None
    created_at: str | None = None
    dataset_path: str | None = None
    model_names: list[str] = Field(default_factory=list)
    evaluator_names: list[str] = Field(default_factory=list)
    error_message: str | None = None
    audit_path: str | None = None
    config_path: str | None = None


class RunListResponse(BaseModel):
    runs: list[RunSummaryResponse]
    count: int


class RunDetailResponse(RunSummaryResponse):
    config_hash: str | None = None
    artifact_paths: dict | None = None
    message: str | None = None


class RunResultsResponse(BaseModel):
    run_id: str
    run_name: str | None = None
    status: str | None = None
    output_dir: str | None = None
    model_results: list[dict] = Field(default_factory=list)
    comparison: dict | None = None


class RunAuditResponse(BaseModel):
    run_id: str
    audit: dict


class ArtifactFileResponse(BaseModel):
    kind: str
    path: str
    exists: bool = True
    model_name: str | None = None


class RunArtifactsResponse(BaseModel):
    run_id: str
    output_dir: str | None = None
    files: list[ArtifactFileResponse] = Field(default_factory=list)
