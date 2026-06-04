from fastapi import APIRouter, Depends

from apps.api.dependencies import AppSettings, get_config_service, get_settings
from apps.api.services.discovery import discover_models
from apps.api.services.response_builders import build_model_list
from apps.api.schemas.models import ModelListResponse
from llm_eval.core.config_service import ConfigService

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=ModelListResponse)
def list_models(
    settings: AppSettings = Depends(get_settings),
    config_service: ConfigService = Depends(get_config_service),
) -> ModelListResponse:
    models, providers = discover_models(settings, config_service)
    return build_model_list(models, providers)
