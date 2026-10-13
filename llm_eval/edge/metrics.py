"""Aggregate latency, throughput, and energy samples. No device access."""

from __future__ import annotations

from typing import Any


def na(reason: str) -> dict:
    return {"value": "n/a", "reason": reason}


def measured(value: float | int, digits: int | None = None) -> dict:
    if isinstance(value, float) and digits is not None:
        value = round(value, digits)
    return {"value": value, "reason": ""}


def metric_number(cell: dict | None) -> float | int | None:
    if not isinstance(cell, dict):
        return None
    value = cell.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return value


def format_metric(cell: dict | None) -> str:
    """Plain text for a metric cell. Missing values read ``n/a`` plus the reason."""
    if not isinstance(cell, dict):
        return "n/a"
    value = cell.get("value")
    if value == "n/a" or value is None:
        reason = str(cell.get("reason") or "").strip()
        return f"n/a ({reason})" if reason else "n/a"
    return str(value)


def percentile(values: list[float], pct: float) -> float:
    """Linear percentile. Same interpolation the latency evaluator uses."""
    data = sorted(values)
    if not data:
        raise ValueError("percentile requires at least one value")
    if len(data) == 1:
        return float(data[0])
    idx = (pct / 100) * (len(data) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(data) - 1)
    frac = idx - lo
    return float(data[lo] + frac * (data[hi] - data[lo]))


def percentile_pair(values: list[float], *, empty_reason: str) -> dict:
    if not values:
        return na(empty_reason)
    return {
        "p50": measured(percentile(values, 50), 2),
        "p95": measured(percentile(values, 95), 2),
        "n": len(values),
    }


def sample_from_response(response, wall_ms: float | None = None) -> dict:
    """One generate call. TTFT stays empty when the connector did not stream."""
    meta = getattr(response, "metadata", None) or {}
    latency = getattr(response, "latency_ms", None)
    if latency is None:
        latency = wall_ms
    ttft = meta.get("ttft_ms")
    ttft_reason = str(meta.get("ttft_reason") or "")
    output = meta.get("output_tokens")
    if output is None and meta.get("completion_tokens") is not None:
        output = meta.get("completion_tokens")
    if output is None and meta.get("prompt_eval_count") is not None:
        output = getattr(response, "tokens_used", None)
    eval_seconds = meta.get("eval_seconds")
    if eval_seconds is None and meta.get("eval_duration_ns"):
        eval_seconds = float(meta["eval_duration_ns"]) / 1e9
    if ttft is None and not ttft_reason:
        # A stream that opened but never sent a token is not a missing connector.
        ttft_reason = "stream returned no token" if meta.get("streaming") else "connector did not stream"
    return {
        "latency_ms": None if latency is None else float(latency),
        "ttft_ms": None if ttft is None else float(ttft),
        "ttft_reason": ttft_reason,
        "output_tokens": None if output is None else int(output),
        "eval_seconds": None if eval_seconds is None else float(eval_seconds),
    }


def tokens_per_second(samples: list[dict]) -> tuple[float | None, str]:
    """Output tokens per second.

    Prefer the server's eval duration. Otherwise use the time after the first
    token. Otherwise use the full request.
    """
    eval_tokens = 0
    eval_seconds = 0.0
    after_tokens = 0
    after_seconds = 0.0
    wall_tokens = 0
    wall_seconds = 0.0
    for sample in samples:
        tokens = sample.get("output_tokens")
        if not tokens:
            continue
        if sample.get("eval_seconds"):
            eval_tokens += int(tokens)
            eval_seconds += float(sample["eval_seconds"])
        latency = sample.get("latency_ms")
        ttft = sample.get("ttft_ms")
        if latency is not None and ttft is not None:
            span = (float(latency) - float(ttft)) / 1000.0
            if span > 0:
                after_tokens += int(tokens)
                after_seconds += span
        if latency:
            wall_tokens += int(tokens)
            wall_seconds += float(latency) / 1000.0
    if eval_seconds > 0:
        return eval_tokens / eval_seconds, "server eval duration"
    if after_seconds > 0:
        return after_tokens / after_seconds, "time after first token"
    if wall_seconds > 0:
        return wall_tokens / wall_seconds, "full request time"
    return None, "no output tokens"


def joules_from_power(samples: list[tuple[float, float]]) -> float | None:
    """Trapezoid integral of watts over seconds. Needs two samples."""
    if len(samples) < 2:
        return None
    ordered = sorted(samples, key=lambda row: row[0])
    total = 0.0
    used = False
    for (t0, w0), (t1, w1) in zip(ordered, ordered[1:]):
        dt = t1 - t0
        if dt <= 0:
            continue
        total += (float(w0) + float(w1)) / 2.0 * dt
        used = True
    return total if used else None


def _ttft_reason(samples: list[dict]) -> str:
    reasons = []
    for sample in samples:
        reason = str(sample.get("ttft_reason") or "").strip()
        if reason and reason not in reasons:
            reasons.append(reason)
    if not samples:
        return "no completed requests"
    return reasons[0] if reasons else "connector did not stream"


def summarize_requests(samples: list[dict], *, concurrency: int, source: str) -> dict:
    latencies = [float(row["latency_ms"]) for row in samples if row.get("latency_ms") is not None]
    ttfts = [float(row["ttft_ms"]) for row in samples if row.get("ttft_ms") is not None]
    output_tokens = sum(int(row["output_tokens"]) for row in samples if row.get("output_tokens"))
    rate, basis = tokens_per_second(samples)
    summary = {
        "concurrency": int(concurrency),
        "source": source,
        "requests": len(samples),
        "latency_ms": percentile_pair(latencies, empty_reason="no completed requests"),
        "ttft_ms": percentile_pair(ttfts, empty_reason=_ttft_reason(samples)),
        "output_tokens": measured(output_tokens, None) if output_tokens else na("no output tokens"),
        "output_tokens_per_second": measured(rate, 2) if rate is not None else na(basis),
        "output_tokens_per_second_basis": basis if rate is not None else "",
    }
    return summary


def summarize_power(
    samples: list[dict],
    *,
    gpu_enabled: bool,
    gpu_reason: str,
    output_tokens: int,
) -> dict:
    """Peak RAM is local. VRAM, watts, and joules need a local NVIDIA GPU."""
    ram_values = [float(row["ram_mib"]) for row in samples if row.get("ram_mib") is not None]
    peak_ram = measured(max(ram_values), 1) if ram_values else na("RAM was not readable")
    if not gpu_enabled:
        reason = gpu_reason or "no NVIDIA GPU detected"
        energy = na(reason)
        return {
            "peak_ram_mib": peak_ram,
            "peak_vram_mib": na(reason),
            "power_watts_average": na(reason),
            "power_watts_peak": na(reason),
            "joules": energy,
            "joules_per_output_token": na(reason),
        }
    vram_values = [float(row["vram_mib"]) for row in samples if row.get("vram_mib") is not None]
    watt_rows = [
        (float(row["t"]), float(row["watts"]))
        for row in samples
        if row.get("watts") is not None and row.get("t") is not None
    ]
    peak_vram = measured(max(vram_values), 1) if vram_values else na("GPU did not report VRAM")
    if watt_rows:
        watts = [row[1] for row in watt_rows]
        average = measured(sum(watts) / len(watts), 2)
        peak = measured(max(watts), 2)
        energy = joules_from_power(watt_rows)
        joules = measured(energy, 2) if energy is not None else na("fewer than two power samples")
    else:
        average = na("GPU did not report power")
        peak = na("GPU did not report power")
        joules = na("GPU did not report power")
    joule_number = metric_number(joules)
    if joule_number is None:
        per_token = na(str(joules.get("reason") or "power was not measured"))
    elif output_tokens <= 0:
        per_token = na("no output tokens")
    else:
        per_token = measured(float(joule_number) / float(output_tokens), 6)
    return {
        "peak_ram_mib": peak_ram,
        "peak_vram_mib": peak_vram,
        "power_watts_average": average,
        "power_watts_peak": peak,
        "joules": joules,
        "joules_per_output_token": per_token,
    }


def combine_summary(
    requests: dict,
    power: dict,
    hardware: dict,
    *,
    serial: dict | None = None,
) -> dict:
    summary = dict(requests)
    summary.update(power)
    summary["hardware"] = hardware
    if serial is not None:
        summary["serial"] = serial
    return summary


def scorecard_edge(summary: dict | None) -> dict:
    """Compact block stored on the scorecard. Category scores are unchanged."""
    summary = summary or {}
    hardware = summary.get("hardware") or {}
    keys = (
        "concurrency",
        "source",
        "requests",
        "ttft_ms",
        "output_tokens_per_second",
        "latency_ms",
        "output_tokens",
        "peak_vram_mib",
        "peak_ram_mib",
        "power_watts_average",
        "power_watts_peak",
        "joules",
        "joules_per_output_token",
    )
    block: dict[str, Any] = {key: summary.get(key) for key in keys}
    block["hardware"] = {
        key: hardware.get(key)
        for key in ("gpu", "driver", "cuda", "cpu", "ram_mib", "os", "power_mode", "endpoint")
    }
    return block
