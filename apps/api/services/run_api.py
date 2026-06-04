"""API-layer helpers for run routes (no business logic)."""

from __future__ import annotations

from apps.api.dependencies import AppSettings
from apps.api.error_handlers import BadRequestError, NotFoundError
from apps.api.schemas.runs import CreateRunRequest
from apps.api.services.response_builders import (
    build_run_artifacts,
    build_run_audit,
    build_run_detail,
    build_run_results,
)
from apps.api.services.run_jobs import RunJob, RunJobManager
from llm_eval.core.config_service import ConfigService
from llm_eval.core.run_service import RunService

IN_PROGRESS_STATUSES = {"pending", "running"}


def is_in_progress(status: str | None) -> bool:
    return status in IN_PROGRESS_STATUSES


def normalize_create_run_request(
    request: CreateRunRequest,
    config_service: ConfigService,
    settings: AppSettings,
) -> tuple[dict, str | None, str]:
    """Validate and normalize a create-run request into an engine config dict."""
    if request.config_path and request.config:
        raise BadRequestError("Provide either config_path or config, not both.")
    if not request.config_path and not request.config:
        raise BadRequestError("Either config_path or config is required.")

    config_source_path: str | None
    try:
        if request.config_path:
            config = config_service.load_yaml(request.config_path)
            config_source_path = request.config_path
        else:
            config = dict(request.config or {})
            config_source_path = None
    except FileNotFoundError as exc:
        raise NotFoundError(str(exc)) from exc
    except ValueError as exc:
        raise BadRequestError(str(exc)) from exc

    if request.run_name:
        config["run_name"] = request.run_name
    if request.output_dir:
        config["output_dir"] = request.output_dir

    structure_errors = config_service.validate_config_structure(config)
    if structure_errors:
        raise BadRequestError("; ".join(structure_errors))

    output_dir = str(config.get("output_dir", settings.output_dir))
    config["output_dir"] = output_dir
    if config_source_path:
        config["_config_path"] = config_source_path

    return config, config_source_path, output_dir


def get_run_entry_or_404(
    run_id: str,
    output_dir: str,
    job_manager: RunJobManager,
) -> dict:
    entry = job_manager.get_merged_run(run_id, output_dir)
    if entry is None:
        raise NotFoundError(f"Run not found: {run_id}")
    return entry


def get_live_job_status(job_manager: RunJobManager, run_id: str) -> str | None:
    live = job_manager.get_live_job(run_id)
    return live.status if live is not None else None


def in_progress_message(status: str, resource: str) -> str:
    return f"Run is still {status}. {resource} will be available when the run completes."


def build_run_detail_response(entry: dict):
    detail = build_run_detail(entry)
    status = entry.get("status")
    in_progress = is_in_progress(status)
    return detail.model_copy(
        update={
            "in_progress": in_progress,
            "ready": not in_progress,
            "message": entry.get("message") or (in_progress_message(status, "Details") if in_progress else None),
        }
    )


def build_run_results_response(
    *,
    run_id: str,
    entry: dict,
    run_service: RunService,
    output_dir: str,
    live_status: str | None,
):
    status = live_status or entry.get("status")
    if is_in_progress(status):
        return build_run_results(
            {
                "run_id": run_id,
                "run_name": entry.get("run_name"),
                "status": status,
                "output_dir": entry.get("output_dir"),
                "model_results": [],
                "comparison": None,
                "ready": False,
                "message": in_progress_message(status, "Results"),
            }
        )

    payload = run_service.get_run_results(run_id, output_dir)
    if payload is None:
        payload = {
            "run_id": run_id,
            "run_name": entry.get("run_name"),
            "status": entry.get("status"),
            "output_dir": entry.get("output_dir"),
            "model_results": [],
            "comparison": None,
        }
    payload["ready"] = True
    payload["message"] = None
    return build_run_results(payload)


def build_run_audit_response(
    *,
    run_id: str,
    entry: dict,
    run_service: RunService,
    output_dir: str,
    live_status: str | None,
):
    status = live_status or entry.get("status")
    if is_in_progress(status):
        return build_run_audit(
            run_id=run_id,
            audit=None,
            ready=False,
            message=in_progress_message(status, "Audit metadata"),
            status=status,
        )

    audit = run_service.get_run_audit(run_id, output_dir)
    if audit is None:
        raise NotFoundError(f"Audit metadata not found for run: {run_id}")

    return build_run_audit(
        run_id=run_id,
        audit=audit.model_dump(mode="json"),
        ready=True,
        message=None,
        status=audit.status,
    )


def build_run_artifacts_response(
    *,
    run_id: str,
    entry: dict,
    run_service: RunService,
    output_dir: str,
    live_status: str | None,
):
    status = live_status or entry.get("status")
    if is_in_progress(status):
        return build_run_artifacts(
            run_id=run_id,
            output_dir=entry.get("output_dir"),
            artifact_paths=None,
            ready=False,
            message=in_progress_message(status, "Artifacts"),
            status=status,
        )

    artifacts = run_service.get_run_artifacts(run_id, output_dir)
    artifact_paths = artifacts.model_dump() if artifacts else entry.get("artifact_paths")
    audit = run_service.get_run_audit(run_id, output_dir)
    if audit and audit.audit_path:
        artifact_paths = dict(artifact_paths or {})
        artifact_paths["audit_path"] = audit.audit_path

    return build_run_artifacts(
        run_id=run_id,
        output_dir=entry.get("output_dir"),
        artifact_paths=artifact_paths,
        ready=True,
        message=None,
        status=entry.get("status"),
    )
