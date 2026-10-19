"""YAML map and resolved-result models.

``extra`` is forbidden so a typo fails validation. Authority is one optional
human sentence. This module does not choose it from the evidence status.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = 1

# Compared case-insensitively. Reports, JSON, and CLI text must not contain it.
FORBIDDEN_AUTHORITY = "Authority granted"

Risk = Literal["low", "moderate", "high", "critical"]
EvidenceStatus = Literal["met", "partial", "gap", "not_assessed"]
AttestationStatus = Literal["met", "partial", "gap"]
ComparisonOp = Literal[">=", ">", "<=", "<"]
AuthorityLevel = Literal[0, 1, 2, 3]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SystemInfo(StrictModel):
    name: str = Field(min_length=1)
    version: str = ""
    owner: str = ""
    summary: str = ""
    fictional: bool = False


class EvidenceFilters(StrictModel):
    """Which click-through runs may supply a metric. Unset fields match any run."""

    preset: str | None = None
    dataset_sha256: str | None = None


class MapDefaults(StrictModel):
    max_result_age_days: int = Field(default=30, ge=0)
    min_samples: int = Field(default=100, ge=1)
    confidence: float = 0.95
    filters: EvidenceFilters = Field(default_factory=EvidenceFilters)

    @field_validator("confidence")
    @classmethod
    def confidence_is_95(cls, value: float) -> float:
        if abs(float(value) - 0.95) > 1e-9:
            raise ValueError("confidence must be 0.95")
        return 0.95


class Attestation(StrictModel):
    """A person's signed statement for a manual evidence item.

    ``signature`` is the mark they recorded. This package does not check a
    cryptographic signature.
    """

    who: str = Field(min_length=1)
    date: date
    signature: str = Field(min_length=1)
    note: str = ""
    expiry: date | None = None
    status: AttestationStatus

    @model_validator(mode="after")
    def expiry_not_before_date(self) -> Attestation:
        if self.expiry is not None and self.expiry < self.date:
            raise ValueError("expiry is before the attestation date")
        return self


class EvidenceRequirement(StrictModel):
    id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    metric: str | None = None
    op: ComparisonOp | None = None
    threshold: float | None = None
    partial_floor: float | None = None
    filters: EvidenceFilters = Field(default_factory=EvidenceFilters)
    min_samples: int | None = Field(default=None, ge=1)
    max_result_age_days: int | None = Field(default=None, ge=0)
    manual: bool = False
    attestation: Attestation | None = None

    @field_validator("threshold", "partial_floor")
    @classmethod
    def finite_bound(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if value != value or value in {float("inf"), float("-inf")}:
            raise ValueError("bounds must be finite")
        return float(value)

    @model_validator(mode="after")
    def manual_or_metric(self) -> EvidenceRequirement:
        if self.manual:
            if self.metric or self.op or self.threshold is not None or self.partial_floor is not None:
                raise ValueError("manual evidence does not take a metric, op, or threshold")
            return self
        if self.attestation is not None:
            raise ValueError("attestation is only for manual evidence")
        if not self.metric or self.op is None or self.threshold is None:
            raise ValueError("scored evidence needs metric, op, and threshold")
        if self.partial_floor is not None:
            higher = self.op in {">=", ">"}
            if higher and self.partial_floor > self.threshold:
                raise ValueError("partial_floor must be at or below the threshold")
            if not higher and self.partial_floor < self.threshold:
                raise ValueError("partial_floor must be at or above the threshold")
        return self


class ModelChangedTrigger(StrictModel):
    kind: Literal["model_changed"]


class DatasetChangedTrigger(StrictModel):
    kind: Literal["dataset_changed"]


class AgeDaysTrigger(StrictModel):
    kind: Literal["age_days"]
    value: int = Field(gt=0)


class MetricDriftTrigger(StrictModel):
    kind: Literal["metric_drift"]
    evidence: str = Field(min_length=1)
    delta: float = Field(gt=0)


class ManualTrigger(StrictModel):
    kind: Literal["manual"]
    text: str = Field(min_length=1)


Trigger = Annotated[
    Union[
        ModelChangedTrigger,
        DatasetChangedTrigger,
        AgeDaysTrigger,
        MetricDriftTrigger,
        ManualTrigger,
    ],
    Field(discriminator="kind"),
]


def trigger_text(trigger: Trigger) -> str:
    """Human line for the page. Triggers are not evaluated in this version."""
    if isinstance(trigger, ManualTrigger):
        return trigger.text
    if isinstance(trigger, AgeDaysTrigger):
        return f"Every {trigger.value} days"
    if isinstance(trigger, ModelChangedTrigger):
        return "Model change"
    if isinstance(trigger, DatasetChangedTrigger):
        return "Dataset change"
    return f"{trigger.evidence} moves by {trigger.delta:g}"


class Decision(StrictModel):
    id: str = Field(min_length=1)
    decision: str = Field(min_length=1)
    owner: str = ""
    ai_role: str = ""
    risk: Risk
    risk_note: str = ""
    required_confidence: str = ""
    recommended_authority: str | None = None
    authority_level: AuthorityLevel | None = None
    limits: str = ""
    evidence: list[EvidenceRequirement] = Field(min_length=1)
    triggers: list[Trigger] = Field(default_factory=list)

    @field_validator("recommended_authority")
    @classmethod
    def blank_authority_is_omitted(cls, value: str | None) -> str | None:
        if value is None:
            return None
        text = value.strip()
        return text or None


class DecisionMap(StrictModel):
    schema_version: Literal[1]
    illustrative: bool = False
    system: SystemInfo
    glossary: str = ""
    defaults: MapDefaults = Field(default_factory=MapDefaults)
    decisions: list[Decision] = Field(min_length=1)


class ResolvedEvidence(StrictModel):
    id: str
    label: str
    status: EvidenceStatus
    reason: str = ""
    manual: bool = False
    metric: str | None = None
    op: ComparisonOp | None = None
    threshold: float | None = None
    partial_floor: float | None = None
    value: float | None = None
    kind: Literal["proportion", "point", "manual"] = "manual"
    sample_count: int | None = None
    successes: int | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    run_id: str | None = None
    model: str | None = None
    preset: str | None = None
    dataset_sha256: str | None = None
    completed_at: str | None = None
    age_days: int | None = None
    attestation_who: str | None = None
    attestation_date: str | None = None
    attestation_expiry: str | None = None


class ResolvedDecision(StrictModel):
    id: str
    decision: str
    status: EvidenceStatus
    reason: str = ""
    recommended_authority: str | None = None
    evidence: list[ResolvedEvidence]


class RunProvenance(StrictModel):
    run_id: str
    model: str | None = None
    precision: str | None = None
    preset: str | None = None
    dataset_sha256: str | None = None
    completed_at: str | None = None


class ResolvedMap(StrictModel):
    schema_version: Literal[1] = 1
    system_name: str
    generated_at: datetime
    as_of: datetime
    confidence: float
    caveat: str
    decisions: list[ResolvedDecision]
    runs: list[RunProvenance] = Field(default_factory=list)


def contains_forbidden(text: str) -> bool:
    return FORBIDDEN_AUTHORITY.casefold() in text.casefold()


def walk_strings(value: object) -> list[str]:
    found: list[str] = []
    if isinstance(value, str):
        found.append(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            found.extend(walk_strings(key))
            found.extend(walk_strings(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            found.extend(walk_strings(item))
    return found
