"""Crib of the garak JSONL records this smoke writes locally.

Entry types and field names follow the public report shape: ``start_run setup``,
``init``, ``attempt``, ``eval``, ``probe_summary``, and ``completion``.
Detector scores use the same polarity as garak (higher means a hit). ``passed``
counts outputs under the 0.5 threshold. This file is not produced by the garak
package.
"""

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SCHEMA_VERSION = "garak-shaped-0.1"

SMOKE_NOTE = (
    "garak-shaped local record for the llm-eval-suite fixture eval crib. "
    "This file is not a garak scan report. "
    "The garak package was not imported. "
    "No model was called. "
    "See docs/garak-fixture-eval.md."
)

ATTEMPT_NEW = 0
ATTEMPT_STARTED = 1
ATTEMPT_COMPLETE = 2
PASS_THRESHOLD = 0.5


class MessageRecord(BaseModel):
    """One turn body, shaped like ``garak.attempt.Message`` as serialized."""

    text: str | None = None
    lang: str | None = None
    data_path: str | None = None
    data_type: Any = None
    data_checksum: str | None = None
    notes: dict[str, Any] = Field(default_factory=dict)


class TurnRecord(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: MessageRecord


class ConversationRecord(BaseModel):
    turns: list[TurnRecord] = Field(default_factory=list)
    notes: dict[str, Any] = Field(default_factory=dict)


class SetupEntry(BaseModel):
    """First JSONL row. Flattened config keys use garak's ``section.field`` form."""

    model_config = ConfigDict(populate_by_name=True)

    entry_type: Literal["start_run setup"] = "start_run setup"
    schema_version: str = SCHEMA_VERSION
    live_llm: bool = False
    live_api: bool = False
    source: str
    notes: str = SMOKE_NOTE
    model_type: str = Field(default="fixture.Canned", alias="plugins.model_type")
    model_name: str = Field(
        default="probe-responses.jsonl", alias="plugins.model_name"
    )
    probe_spec: str = Field(default="", alias="plugins.probe_spec")
    generations: int = Field(default=1, alias="run.generations")
    harness: str = Field(default="probewise", alias="run.harness")
    run_id: str = Field(default="", alias="transient.run_id")
    start_time: str = Field(default="", alias="transient.starttime_iso")
    report_prefix: str = Field(default="smoke", alias="reporting.report_prefix")

    @field_validator("live_llm")
    @classmethod
    def stays_offline(cls, value: bool) -> bool:
        if value:
            raise ValueError("this crib does not call an LLM")
        return value

    @field_validator("live_api")
    @classmethod
    def stays_local(cls, value: bool) -> bool:
        if value:
            raise ValueError("this crib does not call a model API")
        return value

    @field_validator("source")
    @classmethod
    def source_not_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("source is required")
        return value


class InitEntry(BaseModel):
    entry_type: Literal["init"] = "init"
    garak_version: str = "not-installed"
    start_time: str
    run: str


class AttemptEntry(BaseModel):
    """One scored fixture row, shaped like ``Attempt.as_dict()``."""

    entry_type: Literal["attempt"] = "attempt"
    uuid: str
    seq: int
    status: Literal[0, 1, 2]
    probe_classname: str
    probe_params: dict[str, Any] = Field(default_factory=dict)
    targets: list[str] = Field(default_factory=list)
    prompt: ConversationRecord
    outputs: list[MessageRecord | None]
    detector_results: dict[str, list[float | None]]
    notes: dict[str, Any] = Field(default_factory=dict)
    goal: str
    conversations: list[ConversationRecord]
    reverse_translation_outputs: list[Any] = Field(default_factory=list)
    intent: str | None = None

    @field_validator("probe_classname", "goal", "uuid")
    @classmethod
    def text_not_blank(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("attempt identity fields are required")
        return value

    @field_validator("outputs")
    @classmethod
    def one_output_slot(
        cls, value: list[MessageRecord | None]
    ) -> list[MessageRecord | None]:
        if len(value) != 1:
            raise ValueError("this crib records one output slot per attempt")
        return value


class IntentCounts(BaseModel):
    passed: int = Field(ge=0)
    total_evaluated: int = Field(ge=0)
    nones: int = Field(ge=0)


class EvalEntry(BaseModel):
    """One probe/detector total, written after that pair is scored."""

    entry_type: Literal["eval"] = "eval"
    probe: str
    detector: str
    passed: int = Field(ge=0)
    fails: int = Field(ge=0)
    nones: int = Field(ge=0)
    total_evaluated: int = Field(ge=0)
    total_processed: int = Field(ge=0)
    intents: dict[str, IntentCounts] | None = None

    @model_validator(mode="after")
    def counts_add_up(self) -> "EvalEntry":
        if self.passed + self.fails != self.total_evaluated:
            raise ValueError("total_evaluated must equal passed + fails")
        if self.passed + self.fails + self.nones != self.total_processed:
            raise ValueError("total_processed must equal passed + fails + nones")
        return self

    def as_json(self) -> dict[str, Any]:
        payload = self.model_dump(mode="json")
        if payload.get("intents") is None:
            payload.pop("intents")
        return payload


class InferenceCounts(BaseModel):
    total_evaluated: int = Field(ge=0)
    nones: int = Field(ge=0)


class DetectionCounts(BaseModel):
    detectors: list[str]
    passed: int = Field(ge=0)
    fails: int = Field(ge=0)
    nones: int = Field(ge=0)


class ProbeSummaryEntry(BaseModel):
    entry_type: Literal["probe_summary"] = "probe_summary"
    probe: str
    inference_counts: InferenceCounts
    detection_counts: DetectionCounts


class CompletionEntry(BaseModel):
    entry_type: Literal["completion"] = "completion"
    end_time: str
    run: str


class SmokeReport(BaseModel):
    """In-memory report. ``to_jsonl`` is the file garak would have written."""

    setup: SetupEntry
    init: InitEntry
    attempts: list[AttemptEntry]
    evals: list[EvalEntry]
    summaries: list[ProbeSummaryEntry]
    completion: CompletionEntry

    def iter_entries(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = [
            self.setup.model_dump(mode="json", by_alias=True),
            self.init.model_dump(mode="json"),
        ]
        probes = sorted({item.probe_classname for item in self.attempts})
        for probe in probes:
            for attempt in self.attempts:
                if attempt.probe_classname == probe:
                    rows.append(attempt.model_dump(mode="json"))
            for evaluation in self.evals:
                if evaluation.probe == probe:
                    rows.append(evaluation.as_json())
            for summary in self.summaries:
                if summary.probe == probe:
                    rows.append(summary.model_dump(mode="json"))
        rows.append(self.completion.model_dump(mode="json"))
        return rows

    def to_jsonl(self) -> str:
        return "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in self.iter_entries()
        )


def loads_report(text: str) -> SmokeReport:
    """Parse a crib JSONL file back into a report. Rejects a live setup line."""
    raw = [json.loads(line) for line in text.splitlines() if line.strip()]
    if len(raw) < 3:
        raise ValueError("report is missing setup, init, or completion")
    attempts = []
    evals = []
    summaries = []
    for row in raw[2:-1]:
        kind = row.get("entry_type")
        if kind == "attempt":
            attempts.append(AttemptEntry.model_validate(row))
        elif kind == "eval":
            evals.append(EvalEntry.model_validate(row))
        elif kind == "probe_summary":
            summaries.append(ProbeSummaryEntry.model_validate(row))
        else:
            raise ValueError(f"unexpected entry_type: {kind}")
    return SmokeReport(
        setup=SetupEntry.model_validate(raw[0]),
        init=InitEntry.model_validate(raw[1]),
        attempts=attempts,
        evals=evals,
        summaries=summaries,
        completion=CompletionEntry.model_validate(raw[-1]),
    )
