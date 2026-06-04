from pydantic import BaseModel, Field


class DatasetInfo(BaseModel):
    """Metadata about an evaluation dataset."""

    path: str
    format: str = Field(description="File format suffix, e.g. jsonl or csv")
    sample_count: int
    required_fields: list[str] = Field(default_factory=lambda: ["prompt", "expected_answer"])
    optional_fields: list[str] = Field(
        default_factory=lambda: ["category", "difficulty", "metadata"]
    )
