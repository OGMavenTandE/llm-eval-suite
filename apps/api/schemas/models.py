from pydantic import BaseModel, Field


class ModelResponse(BaseModel):
    provider: str
    name: str
    available: bool = False
    connection_status: str = "unknown"
    warning_message: str | None = None
    source_profile: str | None = None


class ModelListResponse(BaseModel):
    models: list[ModelResponse]
    count: int
    providers: list[str] = Field(default_factory=list)
