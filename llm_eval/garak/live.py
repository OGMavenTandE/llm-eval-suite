"""Live garak scan against an OpenAI-compatible endpoint, with a fixture fallback.

RAMPART is not handled here. This module only runs garak when the package is
installed, using ``--target_type openai.OpenAICompatible``. Any other outcome
(not installed, scan error, empty report) returns the canned fixture and sets
``source`` to ``fixture`` so the UI can say so.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

from llm_eval.garak.score import FixtureRow, classify_score, score_detector
from llm_eval.garak.smoke import DEFAULT_FIXTURE, load_fixture

QUICK_PROBES = (
    "dan.Dan_11_0",
    "promptinject.HijackHateHumans",
    "encoding.InjectBase64",
    "leakreplay.LiteratureCloze",
    "lmrc.Profanity",
)


def garak_is_installed() -> bool:
    return importlib.util.find_spec("garak") is not None


def category_for_probe(probe_classname: str) -> str:
    name = (probe_classname or "").lower()
    if any(token in name for token in ("lmrc", "realtoxicityprompts", "continuation")):
        return "toxicity"
    if any(token in name for token in ("misleading", "packagehallucination", "snowball")):
        return "hallucination_factuality"
    return "security_jailbreak"


def build_garak_config(
    *,
    model_name: str,
    base_url: str,
    api_key: str,
    probes: list[str] | str | None,
    generations: int,
    soft_cap: int | None,
) -> dict:
    uri = base_url.rstrip("/") + "/"
    probe_spec = ""
    if isinstance(probes, str):
        probe_spec = "" if probes == "all" else probes
    elif probes:
        probe_spec = ",".join(probes)
    run_block: dict = {"generations": int(generations)}
    if soft_cap:
        run_block["soft_probe_prompt_cap"] = int(soft_cap)
    return {
        "plugins": {
            "target_type": "openai.OpenAICompatible",
            "model_type": "openai.OpenAICompatible",
            "target_name": model_name,
            "model_name": model_name,
            "probe_spec": probe_spec,
        },
        "run": run_block,
        "generators": {
            "openai": {
                "OpenAICompatible": {
                    "uri": uri,
                    "api_key": api_key or "local",
                    "max_tokens": 64,
                }
            }
        },
    }


def build_garak_command(
    *,
    config_path: Path,
    report_prefix: Path,
    model_name: str,
    probes: list[str] | str | None,
    generations: int,
    target_flag: str = "--target_type",
    name_flag: str = "--target_name",
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "garak",
        target_flag,
        "openai.OpenAICompatible",
        name_flag,
        model_name,
        "--generations",
        str(int(generations)),
        "--config",
        str(config_path),
        "--report_prefix",
        str(report_prefix),
    ]
    if isinstance(probes, str) and probes and probes != "all":
        command.extend(["--probes", probes])
    elif isinstance(probes, list) and probes:
        command.extend(["--probes", ",".join(probes)])
    return command


def fixture_items(fixture_path: str | Path | None = None) -> list[dict]:
    rows = load_fixture(fixture_path if fixture_path is not None else DEFAULT_FIXTURE)
    return [_item_from_fixture_row(row) for row in rows]


def _item_from_fixture_row(row: FixtureRow) -> dict:
    scores = {
        name: score_detector(name, row.output, row.triggers) for name in row.detectors
    }
    return _item(
        item_id=f"garak:{row.probe_classname}:{row.seq}",
        probe=row.probe_classname,
        prompt=row.prompt,
        response=row.output,
        detector_scores=scores,
        source="fixture",
    )


def _item(
    *,
    item_id: str,
    probe: str,
    prompt: str,
    response: str | None,
    detector_scores: dict[str, float | None],
    source: str,
) -> dict:
    known = [score for score in detector_scores.values() if score is not None]
    if known:
        hit = sum(known) / len(known)
        passed = all(classify_score(score) == "passed" for score in known)
        # Higher score means the model resisted. Detector hits are inverted.
        safety = 1.0 - hit
    else:
        passed = False
        safety = 0.0
    return {
        "id": item_id,
        "suite": "garak",
        "category": category_for_probe(probe),
        "prompt": prompt,
        "response": response,
        "expected": None,
        "score": round(safety, 4),
        "passed": passed,
        "source": source,
        "detector": ",".join(sorted(detector_scores)),
        "probe": probe,
    }


def parse_garak_report(path: Path) -> list[dict]:
    """Read a garak ``.report.jsonl`` into suite items. Empty if nothing usable."""
    if not path.is_file():
        return []
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError:
            continue
        if entry.get("entry_type") not in (None, "attempt"):
            continue
        if "probe_classname" not in entry and "prompt" not in entry:
            continue
        prompt = _prompt_text(entry.get("prompt"))
        if not prompt:
            continue
        outputs = entry.get("outputs") or []
        response = _output_text(outputs[0]) if outputs else None
        detector_results = entry.get("detector_results") or {}
        scores = {}
        for name, values in detector_results.items():
            if isinstance(values, list) and values:
                raw = values[0]
                scores[name] = None if raw is None else float(raw)
            elif isinstance(values, (int, float)):
                scores[name] = float(values)
        probe = str(entry.get("probe_classname") or "garak.unknown")
        seq = entry.get("seq", len(items))
        items.append(
            _item(
                item_id=f"garak:{probe}:{seq}",
                probe=probe,
                prompt=prompt,
                response=response,
                detector_scores=scores or {"garak.detector": 0.0 if response else None},
                source="live",
            )
        )
    return items


def _prompt_text(prompt) -> str:
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, dict):
        turns = prompt.get("turns") or []
        for turn in reversed(turns):
            content = (turn or {}).get("content")
            if isinstance(content, dict) and content.get("text"):
                return str(content["text"])
            if isinstance(content, str):
                return content
        if prompt.get("text"):
            return str(prompt["text"])
    return ""


def _output_text(output) -> str | None:
    if output is None:
        return None
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        if output.get("text") is not None:
            return str(output["text"])
        content = output.get("content")
        if isinstance(content, dict) and content.get("text") is not None:
            return str(content["text"])
    return str(output)


def run_garak(
    *,
    model_name: str,
    base_url: str,
    api_key: str = "",
    probes: list[str] | str | None = None,
    generations: int = 1,
    soft_cap: int | None = 25,
    work_dir: str | Path,
    timeout_seconds: int = 900,
    cancel_event=None,
    fixture_path: str | Path | None = None,
    runner=None,
) -> dict:
    """Run live garak when it is installed. Otherwise score the fixture.

    ``runner`` replaces ``subprocess.run`` in tests.
    """
    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    if not garak_is_installed():
        return _fixture_result(
            fixture_path,
            notes="garak is not installed. These rows are a fixture. No live model call was made.",
        )

    config = build_garak_config(
        model_name=model_name,
        base_url=base_url,
        api_key=api_key,
        probes=probes,
        generations=generations,
        soft_cap=soft_cap,
    )
    config_path = work / "garak-config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    report_prefix = work / "garak-live"
    env = os.environ.copy()
    env["OPENAICOMPATIBLE_API_KEY"] = api_key or "local"
    env.setdefault("OPENAI_API_KEY", api_key or "local")

    invoke = runner or subprocess.run
    errors: list[str] = []
    completed = None
    for target_flag, name_flag in (
        ("--target_type", "--target_name"),
        ("--model_type", "--model_name"),
    ):
        command = build_garak_command(
            config_path=config_path,
            report_prefix=report_prefix,
            model_name=model_name,
            probes=probes,
            generations=generations,
            target_flag=target_flag,
            name_flag=name_flag,
        )
        if cancel_event is not None and cancel_event.is_set():
            return _fixture_result(
                fixture_path,
                notes="Cancelled before garak started. Showing the fixture.",
            )
        try:
            completed = invoke(
                command,
                cwd=str(work),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            errors.append("garak timed out")
            break
        except Exception as exc:  # noqa: BLE001 - fall back to the fixture
            errors.append(str(exc))
            break
        return_code = getattr(completed, "returncode", 1)
        stderr = getattr(completed, "stderr", "") or ""
        if return_code == 0:
            break
        errors.append(stderr[-500:])
        unrecognized = "unrecognized" in stderr or "target_type" in stderr
        if not unrecognized:
            break

    report_path = Path(str(report_prefix) + ".report.jsonl")
    if not report_path.is_file():
        candidates = list(work.glob("*.report.jsonl"))
        report_path = candidates[0] if candidates else report_path
    items = parse_garak_report(report_path) if report_path.is_file() else []
    if items:
        return {
            "source": "live",
            "label": "Live garak scan",
            "notes": "Scored by garak against the connected endpoint.",
            "items": items,
        }
    detail = errors[-1] if errors else "garak did not write a report"
    return _fixture_result(
        fixture_path,
        notes=(
            "Live garak did not produce a report "
            f"({detail}). These rows are a fixture. No live model result is shown."
        ),
    )


def _fixture_result(fixture_path, *, notes: str) -> dict:
    return {
        "source": "fixture",
        "label": "Fixture (no live model call)",
        "notes": notes,
        "items": fixture_items(fixture_path),
    }
