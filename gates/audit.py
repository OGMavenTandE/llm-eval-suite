"""Gate 3: right-sizing log audit.

Sorts real traffic by task type, reasoning demand, sensitivity, and the cost of
being wrong, and shows which share of calls and tokens could move to a small
model or to plain code. The share is measured on the logs given, never assumed.

The sorting rules live in an audit config (YAML), so a program can tune them
to its own traffic and the rules are part of the record.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from gates import GO, INVALID, NO_GO
from gates.acceptance import Check, GateDecision

ROUTES = ("deterministic", "small", "large")


class TaskType(BaseModel):
    keywords: list[str] = Field(default_factory=list)
    route: str = "small"
    cost_of_wrong: str = "medium"  # low, medium, high

    def model_post_init(self, __context) -> None:
        if self.route not in ROUTES:
            raise ValueError(f"route must be one of {ROUTES}, got {self.route!r}")
        if self.cost_of_wrong not in ("low", "medium", "high"):
            raise ValueError("cost_of_wrong must be low, medium, or high")


class AuditConfig(BaseModel):
    task_types: dict[str, TaskType]
    default_task_type: str = "other"
    reasoning_markers: list[str] = Field(
        default_factory=lambda: ["why", "plan", "compare", "trade-off", "tradeoff", "recommend", "analyze",
                                 "assess", "course of action", "step by step", "explain"]
    )
    long_prompt_tokens: int = 1500  # at or above this, reasoning demand is at least medium
    sensitivity_markers: list[str] = Field(
        default_factory=lambda: [r"\bCUI\b", r"\bFOUO\b", r"\bPII\b", r"\bSSN\b", r"\bNOFORN\b"]
    )
    min_rows: int = 100
    min_classified: float = Field(0.8, ge=0.0, le=1.0)
    # A high-cost task type is held on the large model unless the config routes it explicitly.
    hold_high_cost: bool = True


def load_audit_config(path: str | Path) -> AuditConfig:
    return AuditConfig.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {})


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0


@dataclass
class Sorted:
    task_type: str
    reasoning: str  # low, medium, high
    sensitive: bool
    cost_of_wrong: str
    route: str
    tokens: int
    tokens_estimated: bool


def classify_task(prompt: str, config: AuditConfig) -> str:
    low = prompt.lower()
    best, best_hits = config.default_task_type, 0
    for name, task in config.task_types.items():
        hits = sum(1 for kw in task.keywords if kw.lower() in low)
        if hits > best_hits:
            best, best_hits = name, hits
    return best


def reasoning_demand(prompt: str, tokens: int, config: AuditConfig) -> str:
    low = prompt.lower()
    markers = sum(1 for m in config.reasoning_markers if re.search(rf"\b{re.escape(m)}\b", low))
    if markers >= 2 or (markers >= 1 and tokens >= config.long_prompt_tokens):
        return "high"
    if markers == 1 or tokens >= config.long_prompt_tokens:
        return "medium"
    return "low"


def is_sensitive(prompt: str, config: AuditConfig) -> bool:
    return any(re.search(p, prompt) for p in config.sensitivity_markers)


def route_for(task_type: str, reasoning: str, config: AuditConfig) -> str:
    task = config.task_types.get(task_type)
    if task is None:
        return "large"
    if reasoning == "high" and task.route != "deterministic":
        return "large"
    if config.hold_high_cost and task.cost_of_wrong == "high" and task.route == "small" and reasoning != "low":
        return "large"
    return task.route


def sort_row(row: dict, config: AuditConfig) -> Sorted:
    prompt = str(row.get("prompt") or "")
    if row.get("tokens_in") is not None or row.get("tokens_out") is not None:
        tokens = int(row.get("tokens_in") or 0) + int(row.get("tokens_out") or 0)
        estimated = False
    else:
        tokens = estimate_tokens(prompt) + estimate_tokens(str(row.get("response") or ""))
        estimated = True
    task_type = row.get("task_type") or classify_task(prompt, config)
    reasoning = row.get("reasoning") or reasoning_demand(prompt, tokens, config)
    task = config.task_types.get(task_type)
    return Sorted(
        task_type=task_type,
        reasoning=reasoning,
        sensitive=bool(row.get("sensitive")) if "sensitive" in row else is_sensitive(prompt, config),
        cost_of_wrong=task.cost_of_wrong if task else "unknown",
        route=route_for(task_type, reasoning, config),
        tokens=tokens,
        tokens_estimated=estimated,
    )


@dataclass
class WorkloadRow:
    task_type: str
    calls: int = 0
    tokens: int = 0
    reasoning: dict[str, int] = field(default_factory=lambda: {"low": 0, "medium": 0, "high": 0})
    sensitive: int = 0
    routes: dict[str, int] = field(default_factory=lambda: {r: 0 for r in ROUTES})
    cost_of_wrong: str = "unknown"


def run_audit(rows: list[dict], config: AuditConfig) -> GateDecision:
    table: dict[str, WorkloadRow] = {}
    total_tokens = 0
    any_estimated = False
    route_calls = {r: 0 for r in ROUTES}
    route_tokens = {r: 0 for r in ROUTES}
    for row in rows:
        s = sort_row(row, config)
        entry = table.setdefault(s.task_type, WorkloadRow(s.task_type, cost_of_wrong=s.cost_of_wrong))
        entry.calls += 1
        entry.tokens += s.tokens
        entry.reasoning[s.reasoning] = entry.reasoning.get(s.reasoning, 0) + 1
        entry.sensitive += int(s.sensitive)
        entry.routes[s.route] += 1
        route_calls[s.route] += 1
        route_tokens[s.route] += s.tokens
        total_tokens += s.tokens
        any_estimated |= s.tokens_estimated

    n = len(rows)
    unclassified = table.get(config.default_task_type)
    classified_share = 1 - ((unclassified.calls if unclassified else 0) / n) if n else 0.0
    movable_calls = route_calls["small"] + route_calls["deterministic"]

    decision = GateDecision(3, "Right-sizing log audit", workload="traffic audit")
    decision.checks = [
        Check("Enough traffic to audit", n >= config.min_rows,
              f"{n} log rows; the audit config requires at least {config.min_rows}.", blocks_validity=True),
        Check("Most traffic sorted into a known task type", classified_share >= config.min_classified,
              f"{classified_share * 100:.1f}% of calls matched a task type; the config requires "
              f"{config.min_classified * 100:.1f}%. Add task types or keywords for the rest.",
              blocks_validity=True),
        Check("Some traffic can move off the large model", movable_calls > 0,
              f"{movable_calls} of {n} calls route to a small model or to plain code under these rules."),
    ]
    decision.decide()
    decision.metrics = {
        "rows": n,
        "total_tokens": total_tokens,
        "tokens_estimated": any_estimated,
        "route_calls": route_calls,
        "route_tokens": route_tokens,
        "workload_map": [
            {
                "task_type": e.task_type,
                "calls": e.calls,
                "call_share": e.calls / n if n else 0.0,
                "tokens": e.tokens,
                "token_share": e.tokens / total_tokens if total_tokens else 0.0,
                "reasoning": e.reasoning,
                "sensitive": e.sensitive,
                "routes": e.routes,
                "cost_of_wrong": e.cost_of_wrong,
            }
            for e in sorted(table.values(), key=lambda e: -e.tokens)
        ],
    }
    if any_estimated:
        decision.findings.append(
            "Some rows had no token counts, so their tokens were estimated at four characters per token. "
            "Use logged token counts for a cost decision."
        )
    if decision.verdict == GO and n:
        decision.findings.append(
            f"{route_calls['small']} calls ({route_calls['small'] / n * 100:.1f}%) are candidates for a small "
            f"model and {route_calls['deterministic']} ({route_calls['deterministic'] / n * 100:.1f}%) for plain "
            f"code. These shares come from this log only."
        )
    elif decision.verdict == NO_GO:
        decision.findings.append("Under these rules nothing should move off the large model.")
    elif decision.verdict == INVALID:
        decision.findings.append("Fix the audit inputs before reading the workload map.")
    return decision
