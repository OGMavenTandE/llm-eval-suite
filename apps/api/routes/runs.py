from fastapi import APIRouter, Depends

from apps.api.dependencies import AppSettings, get_config_service, get_run_job_manager, get_run_service, get_settings
from apps.api.schemas.runs import (
    CreateRunRequest,
    CreateRunResponse,
    RunArtifactsResponse,
    RunAuditResponse,
    RunDetailResponse,
    RunListResponse,
    RunResultsResponse,
)
from apps.api.services.response_builders import build_create_run_response, build_run_list
from apps.api.services.run_api import (
    build_run_artifacts_response,
    build_run_audit_response,
    build_run_detail_response,
    build_run_results_response,
    get_live_job_status,
    get_run_entry_or_404,
    normalize_create_run_request,
)
from apps.api.services.run_jobs import RunJobManager
from llm_eval.core.config_service import ConfigService
from llm_eval.core.run_service import RunService

router = APIRouter(prefix="/runs", tags=["runs"])


@router.post("", response_model=CreateRunResponse, status_code=202)
def create_run(
    request: CreateRunRequest,
    settings: AppSettings = Depends(get_settings),
    config_service: ConfigService = Depends(get_config_service),
    job_manager: RunJobManager = Depends(get_run_job_manager),
) -> CreateRunResponse:
    config, _, output_dir = normalize_create_run_request(request, config_service, settings)
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
    return build_run_list(job_manager.list_merged_runs(str(settings.output_dir)))


@router.get("/{run_id}", response_model=RunDetailResponse)
def get_run(
    run_id: str,
    settings: AppSettings = Depends(get_settings),
    job_manager: RunJobManager = Depends(get_run_job_manager),
) -> RunDetailResponse:
    entry = get_run_entry_or_404(run_id, str(settings.output_dir), job_manager)
    return build_run_detail_response(entry)


@router.get("/{run_id}/results", response_model=RunResultsResponse)
def get_run_results(
    run_id: str,
    settings: AppSettings = Depends(get_settings),
    run_service: RunService = Depends(get_run_service),
    job_manager: RunJobManager = Depends(get_run_job_manager),
) -> RunResultsResponse:
    output_dir = str(settings.output_dir)
    entry = get_run_entry_or_404(run_id, output_dir, job_manager)
    return build_run_results_response(
        run_id=run_id,
        entry=entry,
        run_service=run_service,
        output_dir=output_dir,
        live_status=get_live_job_status(job_manager, run_id),
    )


@router.get("/{run_id}/audit", response_model=RunAuditResponse)
def get_run_audit(
    run_id: str,
    settings: AppSettings = Depends(get_settings),
    run_service: RunService = Depends(get_run_service),
    job_manager: RunJobManager = Depends(get_run_job_manager),
) -> RunAuditResponse:
    output_dir = str(settings.output_dir)
    entry = get_run_entry_or_404(run_id, output_dir, job_manager)
    return build_run_audit_response(
        run_id=run_id,
        entry=entry,
        run_service=run_service,
        output_dir=output_dir,
        live_status=get_live_job_status(job_manager, run_id),
    )


@router.get("/{run_id}/artifacts", response_model=RunArtifactsResponse)
def get_run_artifacts(
    run_id: str,
    settings: AppSettings = Depends(get_settings),
    run_service: RunService = Depends(get_run_service),
    job_manager: RunJobManager = Depends(get_run_job_manager),
) -> RunArtifactsResponse:
    output_dir = str(settings.output_dir)
    entry = get_run_entry_or_404(run_id, output_dir, job_manager)
    return build_run_artifacts_response(
        run_id=run_id,
        entry=entry,
        run_service=run_service,
        output_dir=output_dir,
        live_status=get_live_job_status(job_manager, run_id),
    )
