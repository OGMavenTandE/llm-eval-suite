"""Measured seconds per prompt, used to revise preset runtime estimates.

A rate is reused only for the same model, endpoint, and suite. A previous
run of a different model does not set the next estimate.
"""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_SECONDS_PER_PROMPT = 3.0


def timing_key(connection: dict | None, suite: str) -> str:
    connection = connection or {}
    kind = str(connection.get("type") or "").strip()
    endpoint = str(connection.get("base_url") or connection.get("folder") or "").strip().rstrip("/")
    model = str(connection.get("model") or "").strip()
    return "|".join((kind, endpoint, model, suite))


class TimingStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def snapshot(self) -> dict:
        """Unscoped view. Measured rates are not applied across models."""
        return {
            "seconds_per_prompt": DEFAULT_SECONDS_PER_PROMPT,
            "source": "default",
            "prompts": 0,
        }

    def rate_for(self, connection: dict | None, suite: str) -> float | None:
        rates = self._rates()
        row = rates.get(timing_key(connection, suite))
        if not row:
            return None
        try:
            return float(row["seconds_per_prompt"])
        except (KeyError, TypeError, ValueError):
            return None

    def record(self, elapsed_seconds: float, prompts: int, *, connection: dict, suite: str) -> dict:
        if prompts <= 0 or elapsed_seconds < 0:
            current = self.rate_for(connection, suite)
            return {"seconds_per_prompt": current, "prompts": 0, "suite": suite}
        key = timing_key(connection, suite)
        rates = self._rates()
        current = rates.get(key) or {}
        previous_prompts = int(current.get("prompts") or 0)
        previous_rate = current.get("seconds_per_prompt")
        if previous_prompts > 0 and previous_rate is not None:
            total = previous_prompts + prompts
            seconds = (float(previous_rate) * previous_prompts + elapsed_seconds) / total
            prompts_out = total
        else:
            seconds = elapsed_seconds / prompts
            prompts_out = prompts
        rates[key] = {
            "seconds_per_prompt": round(seconds, 4),
            "prompts": prompts_out,
            "suite": suite,
        }
        self._write(rates)
        return rates[key]

    def _rates(self) -> dict:
        if not self.path.is_file():
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
        rates = data.get("rates") if isinstance(data, dict) else None
        if not isinstance(rates, dict):
            return {}
        return dict(rates)

    def _write(self, rates: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"rates": rates}
        self.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
