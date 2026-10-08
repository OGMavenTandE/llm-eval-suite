"""Load the preset file and expand it into the suites a run will execute."""

from __future__ import annotations

import json
from pathlib import Path

from dow_bench.items import count_items as count_dow_items
from llm_eval.garak.live import garak_is_installed, planned_garak_attempts
from llm_eval.offline import demo_base_folder, demo_base_warning, offline_preset_note
from llm_eval_suite.timing import DEFAULT_SECONDS_PER_PROMPT

PRESET_PATH = Path(__file__).with_name("presets.json")

# Measured actual/raw ratios:
# 30.7/35 and 27.8/32, plus 31.8/(26/0.873) and 28.4/(27/0.873).
# The last two raw totals are the shown estimates divided by 0.873.
# The scale is the mean of those four ratios.
MEASURED_ESTIMATE_SCALE = (
    (30.7 / 35.0)
    + (27.8 / 32.0)
    + (31.8 / (26.0 / 0.873))
    + (28.4 / (27.0 / 0.873))
) / 4.0

SUITE_KEYS = (
    "garak",
    "factcheck",
    "robustness",
    "consistency",
    "rampart",
    "dioptra",
    "dow_knowledge",
    "honest_broker",
    "lawful_order",
)

DEMO_MODEL_A_FOLDER = r"C:\AI Eval\LLMs\nanoGPT-master\nanoGPT-master\hf-dow-news"
DEMO_MODEL_B = "gpt2-medium"


def load_presets(path: str | Path | None = None) -> dict:
    preset_path = Path(path) if path is not None else PRESET_PATH
    return json.loads(preset_path.read_text(encoding="utf-8"))


def list_presets(
    path: str | Path | None = None,
    *,
    dataset_rows: int = 50,
    seconds_per_prompt: float | None = None,
    estimate_source: str = "default",
    suite_rates: dict | None = None,
) -> list[dict]:
    presets = load_presets(path)
    rows = []
    for preset_id, body in presets.items():
        estimate = estimate_preset(
            body,
            dataset_rows=dataset_rows,
            seconds_per_prompt=seconds_per_prompt,
            estimate_source=estimate_source,
            suite_rates=suite_rates,
        )
        warning = body.get("warning") or ""
        if body.get("demo"):
            model_b = (body.get("demo") or {}).get("model_b") or {}
            warning = demo_base_warning(demo_base_folder(model_b.get("folder")))
        if body.get("offline") and offline_preset_note() not in warning:
            warning = (warning + " " + offline_preset_note()).strip()
        rows.append(
            {
                "id": preset_id,
                "label": body.get("label", preset_id),
                "description": body.get("description", ""),
                "warning": warning,
                "offline": bool(body.get("offline")),
                "suites": [name for name in SUITE_KEYS if body.get(name)],
                "demo": body.get("demo"),
                **estimate,
            }
        )
    return rows


def _probe_count(garak: dict) -> int | None:
    probes = garak.get("probes")
    if isinstance(probes, list):
        return len(probes)
    if isinstance(probes, str) and probes.strip() and probes.strip() != "all":
        return len([part for part in probes.split(",") if part.strip()])
    if probes == "all":
        return None
    return 0


def estimate_preset(
    body: dict,
    *,
    dataset_rows: int = 50,
    seconds_per_prompt: float | None = None,
    estimate_source: str = "default",
    suite_rates: dict | None = None,
) -> dict:
    """Prompt ceiling and a runtime estimate. Uncapped garak lists stay unestimated.

    ``suite_rates`` maps a suite name to seconds per prompt for this model and
    endpoint. A missing suite rate means the preset time is still estimating.
    ``seconds_per_prompt`` is the single-rate path used when a caller already
    has one rate for every suite.
    """
    rate = DEFAULT_SECONDS_PER_PROMPT if seconds_per_prompt is None else float(seconds_per_prompt)
    source = estimate_source if seconds_per_prompt is not None else "default"
    garak = body.get("garak") or {}
    probe_count = _probe_count(garak) if garak else 0
    cap = garak.get("max_prompts_per_probe") if garak else None
    generations = int(garak.get("generations") or 1) if garak else 1
    unbounded = bool(garak) and (probe_count is None or not cap)
    garak_prompts = None
    if garak and not unbounded:
        # The cap is an upper bound per probe, not the number of prompts that
        # probe will send. Use the probe list when garak can report it.
        # Fact-check, robustness, and consistency are added once below.
        actual = planned_garak_attempts(
            garak.get("probes"),
            cap=int(cap) if cap else None,
            generations=generations,
        )
        if actual is not None:
            garak_prompts = actual
        elif not garak_is_installed() and probe_count is not None and cap:
            garak_prompts = int(probe_count) * int(cap) * generations
        else:
            unbounded = True
    fact = body.get("factcheck") or {}
    fact_prompts = 0
    if fact:
        limit = fact.get("max_items")
        rows = dataset_rows if not limit else min(int(dataset_rows), int(limit))
        fact_prompts = rows * int(fact.get("trials") or 1)
    robust = body.get("robustness") or {}
    robust_prompts = 0
    if robust:
        limit = robust.get("max_items")
        rows = dataset_rows if not limit else min(int(dataset_rows), int(limit))
        perturbations = len(robust.get("perturbations") or ["case"])
        robust_prompts = rows * (1 + perturbations)
    dow_counts = {}
    for dow_name in ("dow_knowledge", "honest_broker", "lawful_order"):
        dow_config = body.get(dow_name)
        if not dow_config:
            continue
        dow_counts[dow_name] = count_dow_items(dow_name, dow_config)
    dow_prompts = sum(dow_counts.values())
    consistency = body.get("consistency") or {}
    consistency_prompts = 0
    if consistency:
        limit = consistency.get("max_items")
        rows = dataset_rows if not limit else min(int(dataset_rows), int(limit))
        consistency_prompts = rows * int(consistency.get("num_runs") or 3)
    known = [
        count
        for count in (garak_prompts, fact_prompts, robust_prompts, consistency_prompts, dow_prompts)
        if count is not None
    ]
    prompt_count = sum(known) if not unbounded else None
    if suite_rates is not None and not unbounded:
        counts = {
            "garak": garak_prompts or 0,
            "factcheck": fact_prompts,
            "robustness": robust_prompts,
            "consistency": consistency_prompts,
            **dow_counts,
        }
        used = []
        total_seconds = 0.0
        missing = False
        for name, count in counts.items():
            if not count:
                continue
            suite_rate = suite_rates.get(name)
            if suite_rate is None:
                missing = True
                break
            used.append(float(suite_rate))
            total_seconds += count * float(suite_rate)
        if missing or not used:
            estimated = None
            source = "estimating"
            rate = None
        else:
            estimated = round(total_seconds * MEASURED_ESTIMATE_SCALE, 1)
            source = "measured"
            rate = (sum(used) / len(used)) * MEASURED_ESTIMATE_SCALE
    else:
        estimated = None if prompt_count is None else round(prompt_count * rate, 1)
        if estimated is None and source != "estimating":
            source = "unbounded"
    return {
        "probe_count": probe_count,
        "prompt_count": prompt_count,
        "factcheck_count": fact_prompts,
        "garak_prompt_count": garak_prompts,
        "estimated_seconds": estimated,
        "seconds_per_prompt": None if rate is None else round(rate, 4),
        "estimate_source": source if estimated is not None else source,
        "suite_rates": dict(suite_rates or {}),
    }


def demo_pair(path: str | Path | None = None, folder: str | None = None) -> dict:
    """Built-in DVIDS fine-tune versus a local gpt2-medium folder. Nothing is downloaded."""
    presets = load_presets(path)
    body = None
    preset_id = None
    for key, value in presets.items():
        if value.get("demo"):
            body = value
            preset_id = key
            break
    if body is None or preset_id is None:
        raise KeyError("No demo preset is defined.")
    demo = dict(body["demo"])
    model_a = dict(demo.get("model_a") or {})
    model_b = dict(demo.get("model_b") or {})
    model_a["folder"] = folder if folder else model_a.get("folder") or DEMO_MODEL_A_FOLDER
    model_b["model"] = model_b.get("model") or DEMO_MODEL_B
    model_b["folder"] = demo_base_folder(model_b.get("folder"))
    model_b["hub"] = False
    return {
        "preset": demo.get("preset") or preset_id,
        "label": body.get("label") or "Demo",
        "model_a": model_a,
        "model_b": model_b,
        "offline": bool(body.get("offline")),
        "base_folder_env": "LLM_EVAL_DEMO_BASE_FOLDER",
    }


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
    warning = body.get("warning") or ""
    if body.get("demo"):
        model_b = (body.get("demo") or {}).get("model_b") or {}
        warning = demo_base_warning(demo_base_folder(model_b.get("folder")))
    if body.get("offline") and offline_preset_note() not in warning:
        warning = (warning + " " + offline_preset_note()).strip()
    return {
        "id": preset_id,
        "label": body.get("label", preset_id),
        "description": body.get("description", ""),
        "warning": warning,
        "offline": bool(body.get("offline")),
        "suites": suites,
    }
