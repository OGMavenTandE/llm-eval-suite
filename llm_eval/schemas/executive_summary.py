from datetime import datetime

from pydantic import BaseModel, Field


class NotableMetric(BaseModel):
    """Human-readable metric highlight for an executive summary."""

    label: str
    value: str
    context: str


class ExecutiveSummary(BaseModel):
    """Plain-language executive summary for a completed evaluation run."""

    run_id: str
    generated_at: datetime
    evaluation_purpose: str
    overall_outcome: str
    recommended_next_step: str
    key_strengths: list[str] = Field(default_factory=list)
    key_weaknesses: list[str] = Field(default_factory=list)
    needs_human_review_count: int = 0
    notable_metrics: list[NotableMetric] = Field(default_factory=list)
    notes: str | None = None


class ReportManifestEntry(BaseModel):
    kind: str
    path: str
    label: str
    content_type: str


class ReportManifest(BaseModel):
    run_id: str
    generated_at: datetime
    artifacts: list[ReportManifestEntry] = Field(default_factory=list)
