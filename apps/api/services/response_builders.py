from datetime import datetime

from apps.api.schemas.common import LinkRef
from apps.api.schemas.datasets import DatasetListResponse, DatasetResponse
from apps.api.schemas.health import HealthResponse, SystemStatusResponse
from apps.api.schemas.models import ModelListResponse, ModelResponse
from apps.api.schemas.profiles import ProfileListResponse, ProfileResponse
from apps.api.schemas.runs import (
    ArtifactFileResponse,
    CreateRunResponse,
    RunArtifactsResponse,
    RunAuditResponse,
    RunDetailResponse,
    RunListResponse,
    RunResultsResponse,
    RunSummaryResponse,
)


def build_health(version: str) -> HealthResponse:
    return HealthResponse(package_version=version)


def build_system_status(payload: dict) -> SystemStatusResponse:
    return SystemStatusResponse(**payload)


def build_profile_list(items: list[dict]) -> ProfileListResponse:
    profiles = [ProfileResponse(**item) for item in items]
    return ProfileListResponse(profiles=profiles, count=len(profiles))


def build_dataset_list(items: list[dict]) -> DatasetListResponse:
    datasets = [DatasetResponse(**item) for item in items]
    return DatasetListResponse(datasets=datasets, count=len(datasets))


def build_model_list(items: list[dict], providers: list[str]) -> ModelListResponse:
    models = [ModelResponse(**item) for item in items]
    return ModelListResponse(models=models, count=len(models), providers=providers)


def build_run_links(run_id: str) -> list[LinkRef]:
    base = f"/runs/{run_id}"
    return [
        LinkRef(rel="self", href=base),
        LinkRef(rel="results", href=f"{base}/results"),
        LinkRef(rel="audit", href=f"{base}/audit"),
        LinkRef(rel="artifacts", href=f"{base}/artifacts"),
    ]


def build_create_run_response(
    *,
    run_id: str,
    status: str,
    output_dir: str,
    dry_run: bool,
    created_at: datetime,
    message: str | None = None,
) -> CreateRunResponse:
    return CreateRunResponse(
        run_id=run_id,
        status=status,
        message=message,
        created_at=created_at,
        output_dir=output_dir,
        dry_run=dry_run,
        links=build_run_links(run_id),
    )


def build_run_summary(entry: dict) -> RunSummaryResponse:
    return RunSummaryResponse(
        run_id=entry["run_id"],
        run_name=entry.get("run_name", entry["run_id"]),
        status=entry.get("status", "unknown"),
        dry_run=entry.get("dry_run", False),
        output_dir=entry.get("output_dir"),
        started_at=entry.get("started_at"),
        completed_at=entry.get("completed_at"),
        created_at=entry.get("created_at"),
        dataset_path=entry.get("dataset_path"),
        model_names=entry.get("model_names") or [],
        evaluator_names=entry.get("evaluator_names") or [],
        error_message=entry.get("error_message"),
        audit_path=entry.get("audit_path"),
        config_path=entry.get("config_path"),
    )


def build_run_list(entries: list[dict]) -> RunListResponse:
    runs = [build_run_summary(entry) for entry in entries]
    return RunListResponse(runs=runs, count=len(runs))


def build_run_detail(entry: dict) -> RunDetailResponse:
    summary = build_run_summary(entry)
    return RunDetailResponse(
        **summary.model_dump(),
        config_hash=entry.get("config_hash"),
        artifact_paths=entry.get("artifact_paths"),
        message=entry.get("message"),
    )


def build_run_results(payload: dict | None) -> RunResultsResponse:
    if payload is None:
        raise ValueError("missing run results")
    return RunResultsResponse(**payload)


def build_run_audit(run_id: str, audit: dict) -> RunAuditResponse:
    return RunAuditResponse(run_id=run_id, audit=audit)


def build_run_artifacts(run_id: str, output_dir: str | None, artifact_paths: dict | None) -> RunArtifactsResponse:
    files: list[ArtifactFileResponse] = []
    if not artifact_paths:
        return RunArtifactsResponse(run_id=run_id, output_dir=output_dir, files=files)

    for model_entry in artifact_paths.get("model_artifacts", []):
        model_name = model_entry.get("model_name")
        for kind, key in (("summary", "summary_path"), ("detailed", "detailed_path")):
            path = model_entry.get(key)
            if path:
                files.append(
                    ArtifactFileResponse(
                        kind=kind,
                        path=path,
                        exists=True,
                        model_name=model_name,
                    )
                )

    for kind, key in (
        ("comparison_summary", "comparison_summary_path"),
        ("comparison_detailed", "comparison_detailed_path"),
    ):
        path = artifact_paths.get(key)
        if path:
            files.append(ArtifactFileResponse(kind=kind, path=path, exists=True))

    audit_path = artifact_paths.get("audit_path")
    if audit_path:
        files.append(ArtifactFileResponse(kind="audit", path=audit_path, exists=True))

    return RunArtifactsResponse(run_id=run_id, output_dir=output_dir, files=files)
