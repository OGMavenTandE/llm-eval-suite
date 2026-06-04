from pydantic import BaseModel, Field


class ModelInfo(BaseModel):
    """Metadata about a configured model."""

    name: str
    provider: str
    params: dict = Field(default_factory=dict)
