"""Read click-through scorecard, edge, and public-benchmark metrics.

Category rates are never averaged. A mixed live/fixture category uses the
live rows when ``items.jsonl`` is present, matching the scorecard. Latency
and other point metrics are returned with no interval.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from llm_eval.decision_map.providers import MetricRegistry, default_registry

# Kept in step with llm_eval_suite.scoring. A unit test checks the sets match.
CATEGORY_KEYS = frozenset(
    {
        "security_jailbreak",
        "toxicity",
        "hallucination_factuality",
        "retrieval",
        "robustness",
    }
)
PUBLIC_SUITES = frozenset(
    {
        "mmlu_pro",
        "gpqa_diamond",
        "gsm8k",
        "math500",
        "ifeval",
        "humaneval_plus",
        "mbpp_plus",
    }
)
EDGE_PERCENTILES = frozenset({"latency_ms", "ttft_ms"})
EDGE_POINTS = frozenset(
    {
        "output_tokens_per_second",
        "peak_vram_mib",
        "peak_ram_mib",
        "joules_per_output_token",
        "power_watts_average",
        "power_watts_peak",
        "output_tokens",
    }
)

Availability = Literal["ok", "not_run", "withheld", "fixture", "absent"]
MetricKind = Literal["proportion", "point"]


@dataclass
class Observation:
    path: str
    kind: MetricKind
    availability: Availability
    value: float | None = None
    sample_count: int | None = None
    successes: int | None = None
    detail: str = ""


class MetricPathError(ValueError):
    def __init__(self, message: str) -> None:
        super().__init__(message)


def successes_from_rate(rate: float, sample_count: int) -> int:
    count = int(round(float(rate) * int(sample_count)))
    return max(0, min(int(sample_count), count))


def declared_kind(path: str) -> MetricKind:
    """Proportion for pass rates. Point for latency, edge totals, and dataset size."""
    family = _builtin(path.split("."))
    if family in {"edge_percentile", "edge_point", "dataset"}:
        return "point"
    return "proportion"


def known_metric_path(path: str, registry: MetricRegistry | None = None) -> None:
    """Raise ``MetricPathError`` when ``path`` is not a built-in or registered path."""
    registry = default_registry if registry is None else registry
    parts = path.split(".")
    if _builtin(parts) is not None:
        return
    if parts and parts[0] == "plugin":
        _plugin(parts, registry)
        return
    raise MetricPathError(f"unknown metric path {path!r}")


def read_metric(
    run: dict,
    items: list[dict] | None,
    path: str,
    registry: MetricRegistry | None = None,
) -> Observation:
    registry = default_registry if registry is None else registry
    parts = path.split(".")
    kind = _builtin(parts)
    if kind == "category":
        return _category(run, items, path, parts[2])
    if kind == "edge_percentile":
        return _edge_percentile(run, path, parts[1], parts[2])
    if kind == "edge_point":
        return _edge_point(run, path, parts[1])
    if kind == "public":
        return _public(run, items, path, parts[1])
    if kind == "dataset":
        return _dataset_count(run, path)
    if parts and parts[0] == "plugin":
        return _plugin_read(run, items, path, parts, registry)
    raise MetricPathError(f"unknown metric path {path!r}")


def _builtin(parts: list[str]) -> str | None:
    if (
        len(parts) == 4
        and parts[0] == "scorecard"
        and parts[1] == "category"
        and parts[3] == "pass_rate"
        and parts[2] in CATEGORY_KEYS
    ):
        return "category"
    if len(parts) == 3 and parts[0] == "edge" and parts[1] in EDGE_PERCENTILES and parts[2] in {"p50", "p95"}:
        return "edge_percentile"
    if len(parts) == 2 and parts[0] == "edge" and parts[1] in EDGE_POINTS:
        return "edge_point"
    if len(parts) == 3 and parts[0] == "public" and parts[2] == "accuracy" and parts[1] in PUBLIC_SUITES:
        return "public"
    if parts == ["dataset", "sample_count"]:
        return "dataset"
    return None


def _plugin(parts: list[str], registry: MetricRegistry) -> None:
    if len(parts) < 3 or not parts[1] or not parts[2]:
        raise MetricPathError(
            "plugin paths look like plugin.<name>.<measurement>, and no outside metrics are built in"
        )
    provider = registry.get(parts[1])
    if provider is None:
        raise MetricPathError(
            f"no metric provider is registered for {parts[1]!r}. "
            "Word error rate and perimeter accuracy are not built in."
        )
    subpath = ".".join(parts[2:])
    if subpath not in provider.paths():
        raise MetricPathError(f"provider {parts[1]!r} does not serve {subpath!r}")


def _number(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        return _number(value.get("value"))
    return None


def _count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not value.is_integer():
        return None
    count = int(value)
    if count < 0:
        return None
    return count


def _unavailable(path: str, kind: MetricKind, availability: Availability, detail: str) -> Observation:
    return Observation(path=path, kind=kind, availability=availability, detail=detail)


def _category(run: dict, items: list[dict] | None, path: str, key: str) -> Observation:
    from_items = _category_from_items(items, path, key)
    if from_items is not None:
        return from_items
    card = run.get("scorecard") or {}
    row = next((item for item in card.get("categories") or [] if item.get("category") == key), None)
    if not isinstance(row, dict):
        return _unavailable(path, "proportion", "absent", "metric absent")
    status = row.get("status")
    source = row.get("source")
    if status == "withheld":
        return _unavailable(path, "proportion", "withheld", "withheld")
    if status == "not_run" or (source is None and row.get("pass_rate") is None):
        return _unavailable(path, "proportion", "not_run", "not run")
    if status == "fixture" or source in {"fixture", "smoke"}:
        return _unavailable(path, "proportion", "fixture", "fixture")
    rate = _number(row.get("pass_rate"))
    sample_count = _count(row.get("sample_count"))
    if rate is None or sample_count is None or sample_count <= 0:
        return _unavailable(path, "proportion", "absent", "metric absent")
    detail = "live rows only" if source == "mixed" else ""
    return Observation(
        path=path,
        kind="proportion",
        availability="ok",
        value=rate,
        sample_count=int(sample_count),
        successes=successes_from_rate(rate, int(sample_count)),
        detail=detail,
    )


def _counts_toward_score(item: dict) -> bool:
    if item.get("counts_toward_score", True) is False:
        return False
    if item.get("category") == "public_benchmark":
        return False
    if item.get("suite") in PUBLIC_SUITES:
        return False
    return True


def _category_from_items(items: list[dict] | None, path: str, key: str) -> Observation | None:
    if not items:
        return None
    matched = [item for item in items if item.get("category") == key and _counts_toward_score(item)]
    if not matched:
        return None
    live = [item for item in matched if item.get("source") == "live"]
    if not live:
        return _unavailable(path, "proportion", "fixture", "fixture")
    successes = sum(1 for item in live if item.get("passed") is True)
    return Observation(
        path=path,
        kind="proportion",
        availability="ok",
        value=successes / len(live),
        sample_count=len(live),
        successes=successes,
        detail="live rows only" if len(live) != len(matched) else "",
    )


def _edge_block(run: dict) -> dict | None:
    card = run.get("scorecard") or {}
    block = card.get("edge")
    if isinstance(block, dict) and block:
        return block
    metrics = run.get("edge_metrics")
    if isinstance(metrics, dict):
        return metrics
    return None


def _edge_n(pair: dict, block: dict) -> int | None:
    count = _count(pair.get("n"))
    if count is not None:
        return count
    return _count(block.get("requests"))


def _edge_percentile(run: dict, path: str, metric: str, tail: str) -> Observation:
    block = _edge_block(run)
    if block is None:
        return _unavailable(path, "point", "absent", "metric absent")
    pair = block.get(metric)
    if not isinstance(pair, dict):
        return _unavailable(path, "point", "absent", "metric absent")
    cell = pair.get(tail)
    value = _number(cell)
    if value is None:
        reason = ""
        if isinstance(cell, dict):
            reason = str(cell.get("reason") or "").strip()
        return _unavailable(path, "point", "absent", reason or "not measured")
    return Observation(
        path=path,
        kind="point",
        availability="ok",
        value=value,
        sample_count=_edge_n(pair, block),
    )


def _edge_point(run: dict, path: str, metric: str) -> Observation:
    block = _edge_block(run)
    if block is None:
        return _unavailable(path, "point", "absent", "metric absent")
    cell = block.get(metric)
    value = _number(cell)
    if value is None:
        reason = ""
        if isinstance(cell, dict):
            reason = str(cell.get("reason") or "").strip()
        return _unavailable(path, "point", "absent", reason or "not measured")
    return Observation(
        path=path,
        kind="point",
        availability="ok",
        value=value,
        sample_count=_count(block.get("requests")),
    )


def _public(run: dict, items: list[dict] | None, path: str, suite_name: str) -> Observation:
    from_items = _public_from_items(items, path, suite_name)
    if from_items is not None:
        return from_items
    for suite in run.get("suites") or []:
        if not isinstance(suite, dict) or suite.get("name") != suite_name:
            continue
        status = suite.get("status")
        source = suite.get("source")
        if status == "not_run":
            return _unavailable(path, "proportion", "not_run", "not run")
        if source in {"fixture", "smoke"}:
            return _unavailable(path, "proportion", "fixture", "fixture")
        metrics = suite.get("metrics") or {}
        rate = _number(metrics.get("accuracy"))
        if rate is None:
            rate = _number(suite.get("pass_rate"))
        sample_count = _count(metrics.get("n_scored"))
        if sample_count is None:
            sample_count = _count(suite.get("done"))
        if rate is None or sample_count is None or sample_count <= 0:
            return _unavailable(path, "proportion", "absent", "metric absent")
        return Observation(
            path=path,
            kind="proportion",
            availability="ok",
            value=rate,
            sample_count=int(sample_count),
            successes=successes_from_rate(rate, int(sample_count)),
        )
    return _unavailable(path, "proportion", "absent", "metric absent")


def _public_from_items(items: list[dict] | None, path: str, suite_name: str) -> Observation | None:
    if not items:
        return None
    matched = [item for item in items if item.get("suite") == suite_name]
    if not matched:
        return None
    live = [item for item in matched if item.get("source") == "live"]
    if not live:
        return _unavailable(path, "proportion", "fixture", "fixture")
    successes = sum(1 for item in live if item.get("passed") is True)
    return Observation(
        path=path,
        kind="proportion",
        availability="ok",
        value=successes / len(live),
        sample_count=len(live),
        successes=successes,
        detail="live rows only" if len(live) != len(matched) else "",
    )


def _dataset_count(run: dict, path: str) -> Observation:
    dataset = run.get("dataset") or {}
    count = _count(dataset.get("row_count"))
    if count is None:
        return _unavailable(path, "point", "absent", "metric absent")
    return Observation(path=path, kind="point", availability="ok", value=float(count), sample_count=count)


def _plugin_read(
    run: dict,
    items: list[dict] | None,
    path: str,
    parts: list[str],
    registry: MetricRegistry,
) -> Observation:
    _plugin(parts, registry)
    provider = registry.get(parts[1])
    assert provider is not None
    subpath = ".".join(parts[2:])
    reading = provider.read(run, list(items or []), subpath)
    if reading is None or reading.value is None:
        detail = reading.detail if reading is not None and reading.detail else "metric absent"
        return _unavailable(path, "point", "absent", detail)
    if reading.source in {"fixture", "smoke"}:
        return _unavailable(path, reading.kind, "fixture", "fixture")
    successes = reading.successes
    if reading.kind == "proportion" and successes is None and reading.sample_count:
        successes = successes_from_rate(float(reading.value), int(reading.sample_count))
    return Observation(
        path=path,
        kind=reading.kind,
        availability="ok",
        value=float(reading.value),
        sample_count=reading.sample_count,
        successes=successes,
        detail=reading.detail,
    )
