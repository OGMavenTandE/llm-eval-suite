"""Load a map file and the click-through runs it may cite."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml
from pydantic import ValidationError

from llm_eval.decision_map.metrics import MetricPathError, known_metric_path
from llm_eval.decision_map.providers import MetricRegistry, default_registry
from llm_eval.decision_map.schema import (
    DecisionMap,
    EvidenceFilters,
    MetricDriftTrigger,
    contains_forbidden,
    walk_strings,
)
from llm_eval.decision_map.status import as_utc


class MapError(Exception):
    def __init__(self, errors: list[str]) -> None:
        self.errors = errors
        super().__init__("\n".join(errors))


@dataclass
class LoadedRun:
    directory: Path
    record: dict
    items: list[dict] | None

    @property
    def run_id(self) -> str:
        recorded = self.record.get("run_id")
        if recorded:
            return str(recorded)
        return self.directory.name

    @property
    def completed_at(self) -> datetime | None:
        raw = self.record.get("completed_at") or self.record.get("created_at")
        return _parse_time(raw)


def load_map(path: Path | str, registry: MetricRegistry | None = None) -> DecisionMap:
    """Parse ``path`` and reject unknown fields, bad metric paths, and forbidden wording."""
    source = Path(path)
    if not source.is_file():
        raise MapError([f"map file not found: {source}"])
    try:
        raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise MapError([f"could not parse YAML: {exc}"]) from exc
    if not isinstance(raw, dict):
        raise MapError(["map file must be a YAML mapping"])
    try:
        decision_map = DecisionMap.model_validate(raw)
    except ValidationError as exc:
        raise MapError(_validation_lines(exc, raw)) from exc
    errors = _semantic_errors(decision_map, raw, registry)
    if errors:
        raise MapError(errors)
    return decision_map


def load_runs(path: Path | str) -> list[LoadedRun]:
    """Read one run directory, one ``run.json``, or a directory of run directories."""
    source = Path(path)
    if not source.exists():
        raise MapError([f"runs path not found: {source}"])
    directories = _run_directories(source)
    return [_load_run(directory) for directory in directories]


def qualifies(run: LoadedRun, filters: EvidenceFilters) -> bool:
    record = run.record
    validity = record.get("validity")
    if validity not in (None, "", "ok"):
        return False
    status = record.get("status") or "completed"
    if status not in {"completed", "ok"}:
        return False
    if filters.preset and record.get("preset") != filters.preset:
        return False
    if filters.dataset_sha256:
        digest = (record.get("dataset") or {}).get("sha256")
        if digest != filters.dataset_sha256:
            return False
    return True


def newest(runs: list[LoadedRun]) -> LoadedRun | None:
    if not runs:
        return None

    def sort_key(run: LoadedRun) -> tuple:
        completed = run.completed_at
        stamp = "" if completed is None else completed.isoformat()
        return (stamp, run.run_id)

    return max(runs, key=sort_key)


def merged_filters(default: EvidenceFilters, item: EvidenceFilters) -> EvidenceFilters:
    return EvidenceFilters(
        preset=item.preset if item.preset is not None else default.preset,
        dataset_sha256=item.dataset_sha256 if item.dataset_sha256 is not None else default.dataset_sha256,
    )


def _semantic_errors(decision_map: DecisionMap, raw: dict, registry: MetricRegistry | None) -> list[str]:
    registry = default_registry if registry is None else registry
    errors: list[str] = []
    for text in walk_strings(raw):
        if contains_forbidden(text):
            errors.append("the map contains wording this tool must not print")
            break
    seen: set[str] = set()
    for decision in decision_map.decisions:
        if decision.id in seen:
            errors.append(f"{decision.id}: duplicate decision id")
        seen.add(decision.id)
        evidence_ids: set[str] = set()
        for evidence in decision.evidence:
            row = f"{decision.id}.{evidence.id}"
            if evidence.id in evidence_ids:
                errors.append(f"{row}: duplicate evidence id")
            evidence_ids.add(evidence.id)
            if evidence.manual or not evidence.metric:
                continue
            try:
                known_metric_path(evidence.metric, registry)
            except MetricPathError as exc:
                errors.append(f"{row}: {exc}")
        for trigger in decision.triggers:
            if isinstance(trigger, MetricDriftTrigger) and trigger.evidence not in evidence_ids:
                errors.append(f"{decision.id}: metric_drift cites unknown evidence {trigger.evidence!r}")
    return errors


def _validation_lines(exc: ValidationError, raw: dict) -> list[str]:
    lines: list[str] = []
    for error in exc.errors():
        loc = error.get("loc") or ()
        row = _row_id(raw, loc)
        place = ".".join(str(part) for part in loc)
        prefix = f"{row}: " if row else ""
        lines.append(f"{prefix}{place}: {error.get('msg')}")
    return lines or ["map file failed validation"]


def _row_id(raw: dict, loc: tuple) -> str:
    decisions = raw.get("decisions")
    if not isinstance(decisions, list) or not loc or loc[0] != "decisions" or len(loc) < 2 or not isinstance(loc[1], int):
        return ""
    index = loc[1]
    if index < 0 or index >= len(decisions) or not isinstance(decisions[index], dict):
        return ""
    decision = decisions[index]
    label = str(decision.get("id") or index)
    if len(loc) >= 4 and loc[2] == "evidence" and isinstance(loc[3], int):
        evidence = decision.get("evidence") or []
        evidence_index = loc[3]
        if isinstance(evidence, list) and 0 <= evidence_index < len(evidence) and isinstance(evidence[evidence_index], dict):
            label = f"{label}.{evidence[evidence_index].get('id') or evidence_index}"
    return label


def _run_directories(source: Path) -> list[Path]:
    if source.is_file():
        if source.name != "run.json":
            raise MapError([f"expected a run.json file, got {source.name}"])
        return [source.parent]
    if (source / "run.json").is_file():
        return [source]
    found = sorted(path.parent for path in source.glob("*/run.json") if path.is_file())
    return found


def _load_run(directory: Path) -> LoadedRun:
    run_path = directory / "run.json"
    try:
        record = json.loads(run_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MapError([f"{run_path}: {exc}"]) from exc
    if not isinstance(record, dict):
        raise MapError([f"{run_path}: run.json must be an object"])
    items_path = directory / "items.jsonl"
    items: list[dict] | None = None
    if items_path.is_file():
        items = []
        for line_number, line in enumerate(items_path.read_text(encoding="utf-8").splitlines(), start=1):
            text = line.strip()
            if not text:
                continue
            try:
                item = json.loads(text)
            except json.JSONDecodeError as exc:
                raise MapError([f"{items_path}:{line_number}: {exc}"]) from exc
            if not isinstance(item, dict):
                raise MapError([f"{items_path}:{line_number}: each line must be an object"])
            items.append(item)
    return LoadedRun(directory=directory, record=record, items=items)


def _parse_time(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return as_utc(parsed)
