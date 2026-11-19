"""Gate 1: the gate spec. What the task is and what "good enough" means.

The spec is written before any model is built or scored. Gates 2, 5 and 6
read their thresholds from it, so a bar cannot move after the results are in
without the spec file changing too.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ClassRule(BaseModel):
    """Extra rules for one label, usually a high-cost one."""

    min_recall: float | None = Field(default=None, ge=0.0, le=1.0)
    cost_of_miss: str = "normal"  # free text shown in reports, e.g. "high: delayed action"


class Acceptance(BaseModel):
    """Bars every candidate must clear. All rates are 0..1."""

    min_accuracy: float = Field(0.0, ge=0.0, le=1.0)
    # The candidate's accuracy may trail the baseline by at most this much,
    # judged on the lower end of a 95% paired bootstrap interval.
    noninferiority_margin: float = Field(0.02, ge=0.0, le=1.0)
    # Share of items that must parse to a valid label. Below it the run is INVALID.
    min_parsed: float = Field(0.95, ge=0.0, le=1.0)
    # Minimum number of held-out items scored. Below it the run is INVALID.
    min_items: int = Field(50, ge=1)
    bootstrap_resamples: int = Field(2000, ge=100)
    seed: int = 20261118


class DriftPolicy(BaseModel):
    """Gate 6 limits, checked after cutover."""

    max_accuracy_drop: float = Field(0.02, ge=0.0, le=1.0)  # on a labeled monitoring set
    max_psi: float = Field(0.2, ge=0.0)  # predicted-label mix, reference against current
    min_agreement: float = Field(0.95, ge=0.0, le=1.0)  # same items, old against new model version


class GateSpec(BaseModel):
    workload: str
    description: str = ""
    labels: list[str]
    classes: dict[str, ClassRule] = Field(default_factory=dict)
    acceptance: Acceptance = Field(default_factory=Acceptance)
    drift: DriftPolicy = Field(default_factory=DriftPolicy)
    prompt_template: str = (
        "Classify the message into exactly one of these labels: {labels}.\n"
        "Reply with the label only.\n\nMessage:\n{text}\n\nLabel:"
    )
    # Boolean item fields to report separately at Gate 5, e.g. heldout_phrasing.
    # Reported for reading, not used as a bar unless a check is added to the spec.
    report_slices: list[str] = Field(default_factory=list)
    owner: str = ""
    baseline_name: str = "baseline"

    @field_validator("labels")
    @classmethod
    def _labels_unique(cls, value: list[str]) -> list[str]:
        if len(value) < 2:
            raise ValueError("a classification workload needs at least two labels")
        if len(set(value)) != len(value):
            raise ValueError("labels must be unique")
        return value

    @model_validator(mode="after")
    def _class_rules_known(self) -> "GateSpec":
        unknown = sorted(set(self.classes) - set(self.labels))
        if unknown:
            raise ValueError(f"class rules name labels that are not in labels: {unknown}")
        return self

    def prompt_for(self, text: str) -> str:
        return self.prompt_template.format(labels=", ".join(self.labels), text=text)


def load_spec(path: str | Path) -> GateSpec:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return GateSpec.model_validate(data)
