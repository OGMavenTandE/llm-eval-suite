from fastapi import APIRouter, Depends

from apps.api.dependencies import AppSettings, get_config_service, get_settings
from apps.api.error_handlers import NotFoundError
from apps.api.schemas.profiles import ProfileDetailResponse, ProfileListResponse
from apps.api.services.discovery import discover_profiles
from apps.api.services.response_builders import build_profile_list
from llm_eval.core.config_service import ConfigService

router = APIRouter(prefix="/profiles", tags=["profiles"])


@router.get("", response_model=ProfileListResponse)
def list_profiles(
    settings: AppSettings = Depends(get_settings),
    config_service: ConfigService = Depends(get_config_service),
) -> ProfileListResponse:
    return build_profile_list(discover_profiles(settings, config_service))


@router.get("/{profile_id}", response_model=ProfileDetailResponse)
def get_profile(
    profile_id: str,
    settings: AppSettings = Depends(get_settings),
    config_service: ConfigService = Depends(get_config_service),
) -> ProfileDetailResponse:
    candidates = [
        settings.config_dir / f"{profile_id}.yaml",
        settings.config_dir / f"{profile_id}.yml",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise NotFoundError(f"Profile not found: {profile_id}")

    try:
        config = config_service.load_yaml(str(path))
        validation = config_service.validate_run(config)
        valid = validation.valid
        validation_message = "; ".join(validation.errors) if validation.errors else None
        evaluators = list(config.get("evaluators", []))
        model_names = [m.name for m in validation.models]
    except Exception as exc:
        valid = False
        validation_message = str(exc)
        evaluators = []
        model_names = []
        config = {}

    return ProfileDetailResponse(
        profile_id=profile_id,
        name=profile_id.replace("_", " ").replace("-", " ").title(),
        path=str(path),
        run_name=config.get("run_name"),
        output_dir=config.get("output_dir", str(settings.output_dir)),
        dataset=config.get("dataset"),
        evaluators=evaluators,
        model_names=model_names,
        valid=valid,
        validation_message=validation_message,
    )
