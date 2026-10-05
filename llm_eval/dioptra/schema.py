"""Crib of the Dioptra resources this pilot records locally.

Field names follow the public Dioptra 1.1.0 docs where the concept is the same
(experiment name and description, metric name/value/step). Identifiers that
Dioptra assigns on a live server (integer group, entrypoint, and job ids) stay
null. This is not a Dioptra server schema.
"""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = "dioptra-shaped-0.1"

PILOT_NOTE = (
    "Dioptra-shaped local record for the llm-eval-suite light pilot. "
    "This file is not a Dioptra server resource. "
    "No REST call was made. "
    "See docs/dioptra-light-pilot.md."
)

# Suite status -> Dioptra job status, only if a later step actually submits.
# A dry-run (validated) is not a job submission.
STATUS_IF_SUBMITTED = {
    "completed": "finished",
    "failed_validation": "failed",
    "failed_runtime": "failed",
}


class MetricRecord(BaseModel):
    """One measurement. Dioptra keys metrics by name plus step."""

    name: str
    value: float | Literal["nan", "inf", "-inf"]
    step: int = 0
    timestamp: datetime | None = None

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("metric name is required")
        return value

    @field_validator("step")
    @classmethod
    def step_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("metric step must be >= 0")
        return value


class JobRecord(BaseModel):
    """Local stand-in for a Dioptra job. submitted is always false in this pilot."""

    local_id: str
    name: str
    entrypoint_name: str
    suite_status: str
    submitted: bool = False
    dioptra_job_id: int | None = None
    dioptra_status_if_submitted: str | None = None
    parameters: dict[str, Any] = Field(default_factory=dict)
    metrics: list[MetricRecord] = Field(default_factory=list)
    metrics_source: str
    artifact_paths: list[str] = Field(default_factory=list)

    @field_validator("submitted")
    @classmethod
    def not_submitted(cls, value: bool) -> bool:
        if value:
            raise ValueError("this pilot does not submit jobs to Dioptra")
        return value


class EntrypointRecord(BaseModel):
    """Local name for a repeatable workflow. Not a registered Dioptra entrypoint."""

    name: str
    description: str
    registered: bool = False
    dioptra_entrypoint_id: int | None = None


class ExperimentRecord(BaseModel):
    """Local stand-in for a Dioptra experiment."""

    name: str
    description: str = ""
    group_id: int | None = None
    group_label: str = "local"
    entrypoints: list[EntrypointRecord] = Field(default_factory=list)
    jobs: list[JobRecord] = Field(default_factory=list)
    created_at: datetime

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("experiment name is required")
        return value

    @field_validator("group_id")
    @classmethod
    def group_unset(cls, value: int | None) -> int | None:
        if value is not None:
            raise ValueError("this pilot does not assign a Dioptra group_id")
        return value


class RestExample(BaseModel):
    """A request body shaped like the public Dioptra REST API. Not sent."""

    method: str
    path: str
    body: dict[str, Any]
    blocked_reason: str
    job_local_id: str | None = None


class PilotBundle(BaseModel):
    """File written by the smoke command and by audit export."""

    schema_version: str = SCHEMA_VERSION
    live_api: bool = False
    source: str
    notes: str = PILOT_NOTE
    warnings: list[str] = Field(default_factory=list)
    experiment: ExperimentRecord
    rest_examples: list[RestExample] = Field(default_factory=list)

    @field_validator("live_api")
    @classmethod
    def stays_offline(cls, value: bool) -> bool:
        if value:
            raise ValueError("this pilot does not contact a Dioptra API")
        return value
