import llm_eval
from fastapi import APIRouter, Depends

from apps.api.dependencies import AppSettings, get_settings
from apps.api.services.discovery import build_system_status
from apps.api.services.response_builders import build_health, build_system_status as build_system_status_response
from apps.api.schemas.health import HealthResponse, SystemStatusResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return build_health(llm_eval.__version__)


@router.get("/system/status", response_model=SystemStatusResponse)
def system_status(settings: AppSettings = Depends(get_settings)) -> SystemStatusResponse:
    return build_system_status_response(build_system_status(settings))
