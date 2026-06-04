from fastapi import APIRouter, Depends

from apps.api.dependencies import AppSettings, get_config_service, get_settings
from apps.api.services.discovery import discover_profiles
from apps.api.services.response_builders import build_profile_list
from apps.api.schemas.profiles import ProfileListResponse
from llm_eval.core.config_service import ConfigService

router = APIRouter(prefix="/profiles", tags=["profiles"])


@router.get("", response_model=ProfileListResponse)
def list_profiles(
    settings: AppSettings = Depends(get_settings),
    config_service: ConfigService = Depends(get_config_service),
) -> ProfileListResponse:
    return build_profile_list(discover_profiles(settings, config_service))
