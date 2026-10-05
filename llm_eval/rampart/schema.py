"""Crib of the RAMPART records this smoke writes locally.

Field names follow the public RAMPART types where the concept is the same
(SafetyStatus values, probe strategy name, harm category strings, tool-call
evidence). This is not a JsonFileReportSink file, and it is not produced by
the rampart package.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

SCHEMA_VERSION = "rampart-shaped-0.1"

SMOKE_NOTE = (
    "RAMPART-shaped local record for the llm-eval-suite pytest smoke crib. "
    "This file is not a RAMPART report sink artifact. "
    "The rampart package was not imported. "
    "No LLM credentials were used. "
    "See docs/rampart-pytest-smoke.md."
)

SafetyStatusValue = Literal["safe", "unsafe", "undetermined", "error"]
EvalOutcomeValue = Literal["detected", "not_detected", "undetermined"]
ObservabilityValue = Literal["tool_and_side_effects", "tool_only", "response_only"]
CaseKind = Literal["evaluator", "probe"]


class ToolCallRecord(BaseModel):
    """A tool invocation on a fixture transcript."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def name_not_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("tool name is required")
        return value


class EvalSignal(BaseModel):
    """Polarity-free detection. Not a safety verdict."""

    outcome: EvalOutcomeValue
    detected: bool
    evidence: list[str] = Field(default_factory=list)
    rationale: str = ""

    @model_validator(mode="after")
    def detected_matches_outcome(self) -> "EvalSignal":
        if self.detected != (self.outcome == "detected"):
            raise ValueError("detected is true only when outcome is detected")
        return self


class TurnRecord(BaseModel):
    """One prompt and the fixture response that answered it."""

    prompt: str
    response_text: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    eval_result: EvalSignal | None = None
    turn_number: int = 0

    @field_validator("turn_number")
    @classmethod
    def turn_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("turn_number must be >= 0")
        return value


class ResultRecord(BaseModel):
    """Local stand-in for a RAMPART Result. safe follows status."""

    status: SafetyStatusValue
    safe: bool
    summary: str
    strategy: str
    observability_level: ObservabilityValue
    harm_category: str | None = None
    turns: list[TurnRecord] = Field(default_factory=list)
    duration_seconds: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def safe_matches_status(self) -> "ResultRecord":
        if self.safe != (self.status == "safe"):
            raise ValueError("safe is true only when status is safe")
        return self

    @field_validator("summary", "strategy")
    @classmethod
    def text_not_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("summary and strategy are required")
        return value


class MarkerNote(BaseModel):
    """The pytest mark the upstream smoke declares. Not registered here."""

    test_name: str
    harm: list[str]

    @field_validator("harm")
    @classmethod
    def harm_not_empty(cls, value: list[str]) -> list[str]:
        if not value or any(not item.strip() for item in value):
            raise ValueError("harm markers need at least one category")
        return value


class SmokeCase(BaseModel):
    """One case from tests/integration/test_smoke.py, replayed locally."""

    name: str
    kind: CaseKind
    harm_category: str
    observability_level: ObservabilityValue
    prompt: str
    response_text: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    watched_tool: str
    manifest_name: str | None = None
    eval_result: EvalSignal | None = None
    result: ResultRecord | None = None

    @model_validator(mode="after")
    def kind_carries_its_record(self) -> "SmokeCase":
        if self.kind == "evaluator":
            if self.eval_result is None:
                raise ValueError("evaluator case requires eval_result")
            if self.result is not None:
                raise ValueError("evaluator case does not record a safety Result")
        if self.kind == "probe" and self.result is None:
            raise ValueError("probe case requires result")
        return self


class SmokeReport(BaseModel):
    """File written by `python -m llm_eval.rampart smoke`."""

    schema_version: str = SCHEMA_VERSION
    live_llm: bool = False
    plugin_registered: bool = False
    source: str
    notes: str = SMOKE_NOTE
    upstream_test: str = "tests/integration/test_smoke.py"
    markers: list[MarkerNote] = Field(default_factory=list)
    cases: list[SmokeCase] = Field(default_factory=list)

    @field_validator("live_llm")
    @classmethod
    def stays_offline(cls, value: bool) -> bool:
        if value:
            raise ValueError("this crib does not call an LLM")
        return value

    @field_validator("plugin_registered")
    @classmethod
    def plugin_stays_off(cls, value: bool) -> bool:
        if value:
            raise ValueError("this crib does not register the RAMPART pytest plugin")
        return value
