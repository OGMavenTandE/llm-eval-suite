from pydantic import BaseModel, Field


class ProfileResponse(BaseModel):
    profile_id: str
    name: str
    description: str | None = None
    path: str
    evaluators: list[str] = Field(default_factory=list)
    model_names: list[str] = Field(default_factory=list)
    available: bool = True
    valid: bool = True
    validation_message: str | None = None


class ProfileListResponse(BaseModel):
    profiles: list[ProfileResponse]
    count: int
