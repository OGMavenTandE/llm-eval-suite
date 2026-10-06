"""Suite runners. Each one implements the same ``run`` shape.

Live garak plugs in through :func:`llm_eval.garak.live.run_garak`. RAMPART stays
a smoke check. Dioptra stays an offline pilot record.
"""

from __future__ import annotations

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


class SuiteRunner(Protocol):
    name: str

    def run(self, ctx: SuiteContext, config: dict) -> dict:
        """Return ``{name, source, label, notes, items}``."""


def _cancelled(ctx: SuiteContext) -> bool:
    return bool(ctx.cancel is not None and ctx.cancel.is_set())


def _emit(ctx: SuiteContext, item: dict) -> None:
    if ctx.on_item is not None:
        ctx.on_item(item)


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


def score_fact(prompt: str, expected: str, response: str) -> tuple[float, bool, str]:
    """Containment, then the existing correctness evaluator."""
    if expected.strip().casefold() in (response or "").casefold():
        evaluator = CorrectnessEvaluator({"mode": "exact_match", "threshold": 1.0})
        result = evaluator.evaluate(prompt, expected, expected)
        return result.score, True, "containment"
    evaluator = CorrectnessEvaluator({"mode": "fuzzy_match", "threshold": 0.8})
    result = evaluator.evaluate(prompt, expected, response or "")
    return result.score, result.passed, result.details.get("mode", "fuzzy_match")


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
                score, passed, mode = score_fact(row["prompt"], row["expected_answer"], result.text)
                item = {
                    "id": item_id,
                    "suite": "factcheck",
                    "category": "hallucination_factuality",
                    "prompt": row["prompt"],
                    "response": result.text,
                    "expected": row["expected_answer"],
                    "score": round(float(score), 4),
                    "passed": bool(passed),
                    "source": "live",
                    "detector": mode,
                    "dataset_category": row.get("category"),
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
        return {
            "name": "garak",
            "source": result["source"],
            "label": result["label"] if result["source"] == "live" else FIXTURE_LABEL,
            "notes": result["notes"],
            "items": items,
            "done": len(result["items"]),
            "total": len(result["items"]),
        }


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
