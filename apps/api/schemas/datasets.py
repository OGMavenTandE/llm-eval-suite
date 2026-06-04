from pydantic import BaseModel, Field


class DatasetResponse(BaseModel):
    dataset_id: str
    name: str
    path: str
    format: str | None = None
    sample_count: int | None = None
    valid: bool = True
    validation_message: str | None = None


class DatasetListResponse(BaseModel):
    datasets: list[DatasetResponse]
    count: int
