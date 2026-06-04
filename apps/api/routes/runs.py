from fastapi import APIRouter, Depends

from apps.api.dependencies import AppSettings, get_config_service, get_run_job_manager, get_run_service, get_settings
from apps.api.error_handlers import BadRequestError, NotFoundError
from apps.api.services.response_builders import (
    build_create_run_response,
    build_run_artifacts,
    build_run_audit,
    build_run_detail,
    build_run_list,
    build_run_results,
)
from apps.api.schemas.runs import (
    CreateRunRequest,
    CreateRunResponse,
    RunArtifactsResponse,
    RunAuditResponse,
    RunDetailResponse,
    RunListResponse,
    RunResultsResponse,
)
from apps.api.services.run_jobs import RunJobManager
from llm_eval.core.config_service import ConfigService
from llm_eval.core.run_service import RunService

router = APIRouter(prefix="/runs", tags=["runs"])


def _resolve_config(request: CreateRunRequest, config_service: ConfigService) -> tuple[dict, str | None]:
    if request.config_path and request.config:
        raise BadRequestError("Provide either config_path or config, not both.")
    if not request.config_path and not request.config:
        raise BadRequestError("Either config_path or config is required.")

    if request.config_path:
        config = config_service.load_yaml(request.config_path)
        config_source_path = request.config_path
    else:
        config = dict(request.config or {})
        config_source_path = None

    if request.run_name:
        config["run_name"] = request.run_name
    if request.output_dir:
        config["output_dir"] = request.output_dir

    return config, config_source_path


@router.post("", response_model=CreateRunResponse, status_code=202)
def create_run(
    request: CreateRunRequest,
    settings: AppSettings = Depends(get_settings),
    config_service: ConfigService = Depends(get_config_service),
    job_manager: RunJobManager = Depends(get_run_job_manager),
) -> CreateRunResponse:
    try:
        config, config_source_path = _resolve_config(request, config_service)
    except FileNotFoundError as exc:
        raise NotFoundError(str(exc)) from exc
    except ValueError as exc:
        raise BadRequestError(str(exc)) from exc

    structure_errors = config_service.validate_config_structure(config)
    if structure_errors:
        raise BadRequestError("; ".join(structure_errors))

    output_dir = str(config.get("output_dir", settings.output_dir))
    config["output_dir"] = output_dir
    if config_source_path:
        config["_config_path"] = config_source_path

    run_id = job_manager.create_run_id()
    run_name = config.get("run_name", "eval_run")
    job = job_manager.submit(
        config,
        dry_run=request.dry_run,
        compare=request.compare,
        run_id=run_id,
        output_dir=output_dir,
        run_name=run_name,
    )

    return build_create_run_response(
        run_id=job.run_id,
        status=job.status,
        output_dir=job.output_dir,
        dry_run=job.dry_run,
        created_at=job.created_at,
        message="Run accepted for local execution.",
    )


@router.get("", response_model=RunListResponse)
def list_runs(
    settings: AppSettings = Depends(get_settings),
    job_manager: RunJobManager = Depends(get_run_job_manager),
) -> RunListResponse:
    entries = job_manager.list_merged_runs(str(settings.output_dir))
    return build_run_list(entries)


@router.get("/{run_id}", response_model=RunDetailResponse)
def get_run(
    run_id: str,
    settings: AppSettings = Depends(get_settings),
    job_manager: RunJobManager = Depends(get_run_job_manager),
) -> RunDetailResponse:
    entry = job_manager.get_merged_run(run_id, str(settings.output_dir))
    if entry is None:
        raise NotFoundError(f"Run not found: {run_id}")
    return build_run_detail(entry)


@router.get("/{run_id}/results", response_model=RunResultsResponse)
def get_run_results(
    run_id: str,
    settings: AppSettings = Depends(get_settings),
    run_service: RunService = Depends(get_run_service),
    job_manager: RunJobManager = Depends(get_run_job_manager),
) -> RunResultsResponse:
    entry = job_manager.get_merged_run(run_id, str(settings.output_dir))
    if entry is None:
        raise NotFoundError(f"Run not found: {run_id}")

    live = job_manager.get_live_job(run_id)
    if live is not None and live.status in {"pending", "running"}:
        raise BadRequestError(
            f"Run '{run_id}' is still {live.status}. Results are not available yet.",
            details={"status": live.status},
        )

    payload = run_service.get_run_results(run_id, str(settings.output_dir))
    if payload is None:
        payload = {
            "run_id": run_id,
            "run_name": entry.get("run_name"),
            "status": entry.get("status"),
            "output_dir": entry.get("output_dir"),
            "model_results": [],
            "comparison": None,
        }
    return build_run_results(payload)


@router.get("/{run_id}/audit", response_model=RunAuditResponse)
def get_run_audit(
    run_id: str,
    settings: AppSettings = Depends(get_settings),
    run_service: RunService = Depends(get_run_service),
    job_manager: RunJobManager = Depends(get_run_job_manager),
) -> RunAuditResponse:
    entry = job_manager.get_merged_run(run_id, str(settings.output_dir))
    if entry is None:
        raise NotFoundError(f"Run not found: {run_id}")

    audit = run_service.get_run_audit(run_id, str(settings.output_dir))
    if audit is None:
        live = job_manager.get_live_job(run_id)
        if live is not None and live.status in {"pending", "running"}:
            raise BadRequestError(
                f"Run '{run_id}' is still {live.status}. Audit metadata is not available yet.",
                details={"status": live.status},
            )
        raise NotFoundError(f"Audit metadata not found for run: {run_id}")

    return build_run_audit(run_id, audit.model_dump(mode="json"))


@router.get("/{run_id}/artifacts", response_model=RunArtifactsResponse)
def get_run_artifacts(
    run_id: str,
    settings: AppSettings = Depends(get_settings),
    run_service: RunService = Depends(get_run_service),
    job_manager: RunJobManager = Depends(get_run_job_manager),
) -> RunArtifactsResponse:
    entry = job_manager.get_merged_run(run_id, str(settings.output_dir))
    if entry is None:
        raise NotFoundError(f"Run not found: {run_id}")

    artifacts = run_service.get_run_artifacts(run_id, str(settings.output_dir))
    artifact_paths = artifacts.model_dump() if artifacts else entry.get("artifact_paths")
    audit = run_service.get_run_audit(run_id, str(settings.output_dir))
    if audit and audit.audit_path:
        artifact_paths = dict(artifact_paths or {})
        artifact_paths["audit_path"] = audit.audit_path

    return build_run_artifacts(run_id, entry.get("output_dir"), artifact_paths)
