"""Measured seconds per prompt, used to revise preset runtime estimates."""

from __future__ import annotations

import json
from pathlib import Path

DEFAULT_SECONDS_PER_PROMPT = 3.0


class TimingStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def snapshot(self) -> dict:
        if not self.path.is_file():
            return {
                "seconds_per_prompt": DEFAULT_SECONDS_PER_PROMPT,
                "source": "default",
                "prompts": 0,
            }
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {
                "seconds_per_prompt": DEFAULT_SECONDS_PER_PROMPT,
                "source": "default",
                "prompts": 0,
            }
        seconds = float(data.get("seconds_per_prompt") or DEFAULT_SECONDS_PER_PROMPT)
        prompts = int(data.get("prompts") or 0)
        source = data.get("source") or ("measured" if prompts else "default")
        return {"seconds_per_prompt": seconds, "source": source, "prompts": prompts}

    def record(self, elapsed_seconds: float, prompts: int) -> dict:
        if prompts <= 0 or elapsed_seconds < 0:
            return self.snapshot()
        current = self.snapshot()
        if current["source"] == "measured" and current["prompts"] > 0:
            total = current["prompts"] + prompts
            seconds = (current["seconds_per_prompt"] * current["prompts"] + elapsed_seconds) / total
            prompts_out = total
        else:
            seconds = elapsed_seconds / prompts
            prompts_out = prompts
        payload = {
            "seconds_per_prompt": round(seconds, 4),
            "source": "measured",
            "prompts": prompts_out,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return payload
