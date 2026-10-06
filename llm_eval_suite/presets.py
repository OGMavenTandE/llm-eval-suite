"""Load the preset file and expand it into the suites a run will execute."""

from __future__ import annotations

import json
from pathlib import Path

PRESET_PATH = Path(__file__).with_name("presets.json")

SUITE_KEYS = ("garak", "factcheck", "robustness", "consistency", "rampart", "dioptra")


def load_presets(path: str | Path | None = None) -> dict:
    preset_path = Path(path) if path is not None else PRESET_PATH
    return json.loads(preset_path.read_text(encoding="utf-8"))


def list_presets(path: str | Path | None = None) -> list[dict]:
    presets = load_presets(path)
    rows = []
    for preset_id, body in presets.items():
        rows.append(
            {
                "id": preset_id,
                "label": body.get("label", preset_id),
                "description": body.get("description", ""),
                "warning": body.get("warning", ""),
                "suites": [name for name in SUITE_KEYS if body.get(name)],
            }
        )
    return rows


def expand_preset(preset_id: str, path: str | Path | None = None) -> dict:
    presets = load_presets(path)
    if preset_id not in presets:
        known = ", ".join(presets)
        raise KeyError(f"Unknown preset '{preset_id}'. Known presets: {known}.")
    body = presets[preset_id]
    suites = []
    for name in SUITE_KEYS:
        config = body.get(name)
        if not config:
            continue
        suites.append({"name": name, **config})
    return {
        "id": preset_id,
        "label": body.get("label", preset_id),
        "description": body.get("description", ""),
        "warning": body.get("warning", ""),
        "suites": suites,
    }
