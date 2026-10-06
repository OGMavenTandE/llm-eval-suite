"""Suite runners. Each one implements the same ``run`` shape.

Live garak plugs in through :func:`llm_eval.garak.live.run_garak`. RAMPART stays
a smoke check. Dioptra stays an offline pilot record.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from llm_eval.datasets.loader import load_dataset
from llm_eval.dioptra.smoke import run_smoke as dioptra_smoke
from llm_eval.evaluators.consistency import ConsistencyEvaluator
from llm_eval.evaluators.correctness import CorrectnessEvaluator
from llm_eval.evaluators.robustness import RobustnessEvaluator
from llm_eval.garak.live import run_garak
from llm_eval.models.base import BaseModel
from llm_eval.models.local_openai_server import LocalOpenAIServer
from llm_eval.rampart.smoke import run_smoke as rampart_smoke
from llm_eval.models.context import strip_think_blocks
from llm_eval_suite.matching import match_expected, normalize_answer_text
from llm_eval_suite.presets import _probe_count

FIXTURE_LABEL = "Fixture / smoke (no live model call)"


@dataclass
class SuiteContext:
    model: BaseModel
    connection: dict
    dataset_rows: list[dict]
    run_dir: Path
    cancel: object
    completed_ids: set[str] = field(default_factory=set)
    on_item: Callable[[dict], None] | None = None
    on_progress: Callable[[int], None] | None = None


class SuiteRunner(Protocol):
    name: str

    def run(self, ctx: SuiteContext, config: dict) -> dict:
        """Return ``{name, source, label, notes, items}``."""


def _cancelled(ctx: SuiteContext) -> bool:
    return bool(ctx.cancel is not None and ctx.cancel.is_set())


def _emit(ctx: SuiteContext, item: dict) -> None:
    if ctx.on_item is not None:
        ctx.on_item(item)


def planned_suite_total(name: str, config: dict, row_count: int) -> int | None:
    """How many scored items this suite will emit. None when the count is not known yet."""
    rows = int(row_count)
    if name in {"factcheck", "robustness", "consistency"}:
        limit = config.get("max_items")
        if limit:
            rows = min(rows, int(limit))
        if name == "factcheck":
            return rows * max(1, int(config.get("trials") or 1))
        return rows
    if name == "garak":
        cap = config.get("max_prompts_per_probe")
        if not cap:
            return None
        count = _probe_count({"probes": config.get("probes")})
        if not count:
            return None
        return int(count) * int(cap) * int(config.get("generations") or 1)
    if name == "dioptra":
        return 1
    return None


def validate_factcheck_text(filename: str, text: str, destination: Path) -> dict:
    """Validate an uploaded CSV or JSONL fact-check file and store it if valid."""
    suffix = Path(filename or "upload.jsonl").suffix.lower()
    if suffix not in {".jsonl", ".csv"}:
        return {
            "ok": False,
            "errors": ["Upload a .jsonl or .csv file with prompt and expected_answer columns."],
            "row_count": 0,
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    try:
        rows = load_dataset(str(destination))
    except (ValueError, FileNotFoundError) as exc:
        destination.unlink(missing_ok=True)
        return {"ok": False, "errors": [str(exc)], "row_count": 0}
    for index, row in enumerate(rows, start=1):
        category = row.get("category")
        source = row.get("source")
        if category is not None and not isinstance(category, str):
            destination.unlink(missing_ok=True)
            return {
                "ok": False,
                "errors": [f"Row {index}: category must be text."],
                "row_count": 0,
            }
        if source is not None and not isinstance(source, str):
            destination.unlink(missing_ok=True)
            return {
                "ok": False,
                "errors": [f"Row {index}: source must be text."],
                "row_count": 0,
            }
    return {"ok": True, "errors": [], "row_count": len(rows), "path": str(destination)}


def score_fact_detail(prompt: str, expected: str, response: str) -> dict:
    """Word-boundary containment, then the existing correctness evaluator.

    ``Newport`` inside ``Newport News`` is a partial match and does not pass.
    """
    expected = normalize_answer_text(expected)
    response = normalize_answer_text(response)
    found = match_expected(expected, response)
    evidence = {
        "span": found.get("span") or "",
        "excerpt": found.get("excerpt") or "",
        "match": found.get("kind") or "none",
    }
    if found["kind"] == "full":
        evaluator = CorrectnessEvaluator({"mode": "exact_match", "threshold": 1.0})
        result = evaluator.evaluate(prompt, expected, expected)
        return {
            "score": result.score,
            "passed": True,
            "mode": "containment",
            "evidence": evidence,
        }
    if found["kind"] == "partial":
        return {"score": 0.0, "passed": False, "mode": "partial", "evidence": evidence}
    evaluator = CorrectnessEvaluator({"mode": "fuzzy_match", "threshold": 0.8})
    result = evaluator.evaluate(prompt, expected, response or "")
    if result.passed and not evidence["excerpt"]:
        evidence["excerpt"] = (response or "")[:160]
    evidence["match"] = evidence["match"] if evidence["match"] != "none" else result.details.get("mode", "fuzzy_match")
    return {
        "score": result.score,
        "passed": result.passed,
        "mode": result.details.get("mode", "fuzzy_match"),
        "evidence": evidence,
    }


def score_fact(prompt: str, expected: str, response: str) -> tuple[float, bool, str]:
    """Containment, then the existing correctness evaluator."""
    detail = score_fact_detail(prompt, expected, response)
    return detail["score"], detail["passed"], detail["mode"]


class FactcheckRunner:
    name = "factcheck"

    def run(self, ctx: SuiteContext, config: dict) -> dict:
        limit = config.get("max_items")
        rows = list(ctx.dataset_rows)
        if limit:
            rows = rows[: int(limit)]
        trials = max(1, int(config.get("trials") or 1))
        items = []
        total = len(rows) * trials
        done = 0
        for trial in range(trials):
            for index, row in enumerate(rows):
                item_id = f"factcheck:{trial}:{index}"
                if item_id in ctx.completed_ids:
                    done += 1
                    continue
                if _cancelled(ctx):
                    return _partial("factcheck", items, total, done, source="live", label="Live fact-check")
                prompt = (
                    "Answer with only the fact. Do not add explanation.\n"
                    f"Question: {row['prompt']}"
                )
                result = ctx.model.generate(prompt, max_tokens=64)
                answer = strip_think_blocks(result.text)
                detail = score_fact_detail(row["prompt"], row["expected_answer"], answer)
                item = {
                    "id": item_id,
                    "suite": "factcheck",
                    "category": "hallucination_factuality",
                    "prompt": row["prompt"],
                    "response": answer,
                    "empty": not bool(answer.strip()),
                    "expected": row["expected_answer"],
                    "score": round(float(detail["score"]), 4),
                    "passed": bool(detail["passed"]),
                    "source": "live",
                    "detector": detail["mode"],
                    "dataset_category": row.get("category"),
                    "evidence": detail["evidence"],
                }
                items.append(item)
                _emit(ctx, item)
                done += 1
        return {
            "name": "factcheck",
            "source": "live",
            "label": "Live fact-check",
            "notes": "Scored with the correctness evaluator against the selected model.",
            "items": items,
            "done": done,
            "total": total,
        }


class RobustnessRunner:
    name = "robustness"

    def run(self, ctx: SuiteContext, config: dict) -> dict:
        limit = config.get("max_items")
        rows = list(ctx.dataset_rows)
        if limit:
            rows = rows[: int(limit)]
        evaluator = RobustnessEvaluator(
            {"perturbations": config.get("perturbations") or ["case"], "threshold": 0.8}
        )
        correctness = CorrectnessEvaluator({"mode": "fuzzy_match", "threshold": 0.8})
        items = []
        for index, row in enumerate(rows):
            item_id = f"robustness:{index}"
            if item_id in ctx.completed_ids:
                continue
            if _cancelled(ctx):
                return _partial("robustness", items, len(rows), index, source="live", label="Live robustness")
            prompt = row["prompt"]
            baseline = ctx.model.generate(prompt, max_tokens=64)
            result = evaluator.evaluate(
                prompt,
                row["expected_answer"],
                baseline.text,
                model=ctx.model,
                correctness_evaluator=correctness,
            )
            item = {
                "id": item_id,
                "suite": "robustness",
                "category": "robustness",
                "prompt": prompt,
                "response": baseline.text,
                "expected": row["expected_answer"],
                "score": round(float(result.score), 4),
                "passed": bool(result.passed),
                "source": "live",
                "detector": "robustness",
            }
            items.append(item)
            _emit(ctx, item)
        return {
            "name": "robustness",
            "source": "live",
            "label": "Live robustness",
            "notes": "Perturbation consistency from the existing robustness evaluator.",
            "items": items,
            "done": len(items),
            "total": len(rows),
        }


class ConsistencyRunner:
    name = "consistency"

    def run(self, ctx: SuiteContext, config: dict) -> dict:
        limit = config.get("max_items")
        rows = list(ctx.dataset_rows)
        if limit:
            rows = rows[: int(limit)]
        evaluator = ConsistencyEvaluator(
            {"num_runs": int(config.get("num_runs") or 3), "threshold": 0.8}
        )
        items = []
        for index, row in enumerate(rows):
            item_id = f"consistency:{index}"
            if item_id in ctx.completed_ids:
                continue
            if _cancelled(ctx):
                return _partial("consistency", items, len(rows), index, source="live", label="Live consistency")
            prompt = row["prompt"]
            first = ctx.model.generate(prompt, max_tokens=64)
            result = evaluator.evaluate(prompt, row["expected_answer"], first.text, model=ctx.model)
            item = {
                "id": item_id,
                "suite": "consistency",
                "category": "robustness",
                "prompt": prompt,
                "response": first.text,
                "expected": row["expected_answer"],
                "score": round(float(result.score), 4),
                "passed": bool(result.passed),
                "source": "live",
                "detector": "consistency",
            }
            items.append(item)
            _emit(ctx, item)
        return {
            "name": "consistency",
            "source": "live",
            "label": "Live consistency",
            "notes": "Repeated answers compared by the existing consistency evaluator.",
            "items": items,
            "done": len(items),
            "total": len(rows),
        }


class GarakRunner:
    name = "garak"

    def run(self, ctx: SuiteContext, config: dict) -> dict:
        if _cancelled(ctx):
            return _partial("garak", [], 0, 0, source="fixture", label=FIXTURE_LABEL)
        work = ctx.run_dir / "garak"
        endpoint = ctx.connection.get("base_url") or ""
        server = None
        if ctx.connection.get("type") == "hf":
            server = LocalOpenAIServer(ctx.model)
            endpoint = server.start()
        max_context = ctx.connection.get("max_context")
        if ctx.connection.get("type") == "hf" and not max_context:
            max_context = 1024
        skip = _completed_probes(work)
        try:
            result = run_garak(
                model_name=ctx.connection.get("model") or "model",
                base_url=endpoint or "http://127.0.0.1:9/v1",
                api_key=ctx.connection.get("api_key") or "",
                probes=config.get("probes"),
                generations=int(config.get("generations") or 1),
                soft_cap=config.get("max_prompts_per_probe"),
                work_dir=work,
                cancel_event=ctx.cancel,
                max_context=int(max_context) if max_context else None,
                max_new_tokens=int(ctx.connection.get("max_new_tokens") or 64),
                mode=_garak_mode(ctx.connection),
                log_path=ctx.run_dir / "run.log",
                skip_probes=skip,
                on_progress=ctx.on_progress,
            )
        finally:
            if server is not None:
                server.stop()
        items = []
        for item in result["items"]:
            if item["id"] in ctx.completed_ids:
                continue
            items.append(item)
            _emit(ctx, item)
        summary = {
            "name": "garak",
            "source": result["source"],
            "label": FIXTURE_LABEL if result["source"] == "fixture" else result["label"],
            "notes": result["notes"],
            "items": items,
            "done": len(result["items"]),
            "total": len(result["items"]),
        }
        for key in (
            "validity",
            "validity_reason",
            "empty_generations",
            "attack_success_rate",
            "pass_rate",
            "pass_rate_label",
            "wording",
            "garak_runs_dir",
            "report_dir",
            "log_path",
        ):
            if key in result:
                summary[key] = result[key]
        return summary


class RampartRunner:
    name = "rampart"

    def run(self, ctx: SuiteContext, config: dict) -> dict:
        if _cancelled(ctx):
            return _partial("rampart", [], 0, 0, source="smoke", label=FIXTURE_LABEL)
        report, _path = rampart_smoke(ctx.run_dir / "rampart-smoke.json")
        items = []
        for index, case in enumerate(report.cases):
            safe = True
            if case.result is not None:
                safe = bool(case.result.safe)
            elif case.eval_result is not None:
                safe = bool(case.eval_result.detected)
            item = {
                "id": f"rampart:{index}",
                "suite": "rampart",
                "category": "security_jailbreak",
                "prompt": case.prompt,
                "response": case.response_text,
                "expected": None,
                "score": 1.0 if safe else 0.0,
                "passed": safe,
                "source": "smoke",
                "detector": case.harm_category,
                "counts_toward_score": True,
            }
            if item["id"] not in ctx.completed_ids:
                items.append(item)
                _emit(ctx, item)
        return {
            "name": "rampart",
            "source": "smoke",
            "label": FIXTURE_LABEL,
            "notes": "RAMPART smoke check. No live model call was made.",
            "items": items,
            "done": len(report.cases),
            "total": len(report.cases),
        }


class DioptraRunner:
    name = "dioptra"

    def run(self, ctx: SuiteContext, config: dict) -> dict:
        if _cancelled(ctx):
            return _partial("dioptra", [], 0, 0, source="fixture", label=FIXTURE_LABEL)
        _bundle, path = dioptra_smoke(ctx.run_dir / "dioptra-record.json")
        item = {
            "id": "dioptra:record",
            "suite": "dioptra",
            "category": "record",
            "prompt": "Dioptra pilot record",
            "response": f"Wrote {path.name}",
            "expected": None,
            "score": None,
            "passed": True,
            "source": "fixture",
            "detector": "dioptra-record",
            "counts_toward_score": False,
        }
        items = []
        if item["id"] not in ctx.completed_ids:
            items.append(item)
            _emit(ctx, item)
        return {
            "name": "dioptra",
            "source": "fixture",
            "label": FIXTURE_LABEL,
            "notes": "Dioptra pilot record from the offline crib. These numbers are not measurements of this model.",
            "items": items,
            "done": 1,
            "total": 1,
        }


def _garak_mode(connection: dict) -> str:
    mode = (connection.get("mode") or "auto").lower()
    if mode in {"completions", "base", "completion"}:
        return "completions"
    if connection.get("type") == "hf" and mode != "chat":
        return "completions"
    return "chat"


def _completed_probes(work: Path) -> list[str]:
    path = work / "completed_probes.json"
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if isinstance(payload, list):
        return [str(item) for item in payload]
    return []


def _partial(name: str, items: list[dict], total: int, done: int, *, source: str, label: str) -> dict:
    return {
        "name": name,
        "source": source,
        "label": label,
        "notes": "Stopped because the run was cancelled.",
        "items": items,
        "done": done,
        "total": total,
        "cancelled": True,
    }


RUNNERS: dict[str, SuiteRunner] = {
    "factcheck": FactcheckRunner(),
    "robustness": RobustnessRunner(),
    "consistency": ConsistencyRunner(),
    "garak": GarakRunner(),
    "rampart": RampartRunner(),
    "dioptra": DioptraRunner(),
}


def seed_robustness(seed: int = 0) -> None:
    random.seed(seed)
