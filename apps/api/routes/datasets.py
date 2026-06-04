from fastapi import APIRouter, Depends

from apps.api.dependencies import AppSettings, get_settings
from apps.api.services.discovery import discover_datasets
from apps.api.services.response_builders import build_dataset_list
from apps.api.schemas.datasets import DatasetListResponse

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.get("", response_model=DatasetListResponse)
def list_datasets(settings: AppSettings = Depends(get_settings)) -> DatasetListResponse:
    return build_dataset_list(discover_datasets(settings))
