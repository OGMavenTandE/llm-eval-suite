from datetime import datetime

from pydantic import BaseModel, Field

from apps.api.schemas.common import TimestampedResponse


class HealthResponse(TimestampedResponse):
    app_status: str = "ok"
    package_version: str
    api_status: str = "ok"


class SystemStatusResponse(TimestampedResponse):
    app_version: str
    output_dir: str
    run_index_path: str
    profiles_count: int
    datasets_count: int
    model_providers: list[str]
    models_count: int
    warnings: list[str] = Field(default_factory=list)
