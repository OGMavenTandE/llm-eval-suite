"""Run one public benchmark suite through the shared model hooks.

Generation uses the same prompt budget, thinking cap, and answer cap as
``dow_bench``. Public items set ``counts_toward_score`` false and
``counts_toward_dow_score`` false so they stay out of the AIEval DoW Bench Score.
"""

from __future__ import annotations

import os
from pathlib import Path

from llm_eval.models.context import strip_think_blocks
from llm_eval.offline import activate_run_offline

from standard_bench import COUNTS_TOWARD_DOW_SCORE, SUITE_VERSION
from standard_bench.catalog import suite_spec
from standard_bench.load import load_file, load_suite, select_rows
from standard_bench.manifest import HashMismatch, ManifestError
from standard_bench.protocol import GPQA_SUITE, QUICK_SEED, protocol_name, subset_limits
from standard_bench.reference import exclusions_for
from standard_bench.sandbox import (
    DEFAULT_CPU_S,
    DEFAULT_MEMORY_MB,
    DEFAULT_TIMEOUT_S,
    RunLimits,
    describe_isolation,
)
from standard_bench.scorers.code import humaneval_prompt, mbpp_prompt, score_code_task
from standard_bench.scorers.gpqa import build_prompt as gpqa_prompt
from standard_bench.scorers.gpqa import score_gpqa, shuffle_options
from standard_bench.scorers.gsm8k import build_prompt as gsm_prompt
from standard_bench.scorers.gsm8k import score_gsm8k
from standard_bench.scorers.ifeval import aggregate_ifeval, score_ifeval_item
from standard_bench.scorers.math500 import build_prompt as math_prompt
from standard_bench.scorers.math500 import score_math
from standard_bench.scorers.mmlu_pro import build_prompt as mmlu_prompt
from standard_bench.scorers.mmlu_pro import score_mmlu_pro

from dow_bench.meta import DEFAULT_ANSWER_CAP, derive_max_context, thinking_cap

LABELS = {name: suite_spec(name).title for name in (
    "mmlu_pro",
    "gpqa_diamond",
    "gsm8k",
    "math500",
    "ifeval",
    "humaneval_plus",
    "mbpp_plus",
)}


def _cancelled(ctx) -> bool:
    return bool(ctx.cancel is not None and ctx.cancel.is_set())


def parse_limits(text: str) -> dict[str, int]:
    parsed: dict[str, int] = {}
    for part in (text or "").split(","):
        piece = part.strip()
        if not piece or "=" not in piece:
            continue
        name, raw = piece.split("=", 1)
        parsed[name.strip()] = int(raw.strip())
    return parsed


def resolve_dest(config: dict) -> Path:
    if config.get("fixtures") or os.environ.get("LLM_EVAL_STANDARD_FIXTURES") == "1":
        explicit = config.get("data_dir") or os.environ.get("LLM_EVAL_STANDARD_DATA")
        if not explicit:
            return Path(__file__).resolve().parent / "fixtures"
    raw = config.get("data_dir") or os.environ.get("LLM_EVAL_STANDARD_DATA") or ""
    if not str(raw).strip():
        raise ManifestError(
            "No standard benchmark directory. Set LLM_EVAL_STANDARD_DATA or pass "
            "--data. Fetch it on a connected machine with "
            "python -m standard_bench fetch --dest <dir>. Nothing was downloaded."
        )
    return Path(str(raw))


def resolve_subset(config: dict) -> str:
    """CLI ``--subset`` wins. Otherwise the preset's subset, or ``full``."""
    env = (os.environ.get("LLM_EVAL_STANDARD_SUBSET") or "").strip()
    name = env or str(config.get("subset") or "full").strip() or "full"
    subset_limits(name)
    return name


def resolve_limit(config: dict, suite: str) -> int | None:
    raw_limits = config.get("limits")
    if isinstance(raw_limits, str):
        limits = parse_limits(raw_limits)
    elif isinstance(raw_limits, dict):
        limits = {str(key): int(value) for key, value in raw_limits.items()}
    else:
        limits = parse_limits(os.environ.get("LLM_EVAL_STANDARD_LIMITS") or "")
    if suite in limits:
        return int(limits[suite])
    if config.get("limit") not in (None, ""):
        return int(config["limit"])
    env_limit = os.environ.get("LLM_EVAL_STANDARD_LIMIT")
    if env_limit:
        return int(env_limit)
    named = subset_limits(resolve_subset(config))
    if named and suite in named:
        return int(named[suite])
    return None


def resolve_seed(config: dict) -> int | None:
    """An explicit seed wins. quick50 uses the fixed seed when none was passed."""
    env_seed = os.environ.get("LLM_EVAL_STANDARD_SEED")
    if env_seed not in (None, ""):
        return int(env_seed)
    if resolve_subset(config) == "quick50":
        if config.get("seed") not in (None, ""):
            return int(config["seed"])
        return QUICK_SEED
    return None


def resolve_thinking_budget(model_name: str | None, config: dict) -> tuple[int, int]:
    """Return ``(applied, configured)``.

    A non-thinking model applies 0. The configured cap is still recorded so a
    quick check shows 512 even when that cap does not bind.
    """
    native = thinking_cap(model_name)
    env = os.environ.get("LLM_EVAL_STANDARD_THINKING_BUDGET")
    if env not in (None, ""):
        configured = max(0, int(env))
    elif config.get("thinking_budget") not in (None, ""):
        configured = max(0, int(config["thinking_budget"]))
    else:
        configured = native
    applied = configured if native > 0 else 0
    return applied, configured


def _protocol_fields(config: dict) -> dict:
    subset = resolve_subset(config)
    seed = resolve_seed(config)
    return {
        "preset": str(config.get("preset") or ""),
        "subset": subset,
        "protocol": protocol_name(subset),
        "seed": seed,
    }


def _answer_cap(ctx, config: dict, suite: str) -> int:
    if ctx.connection.get("max_new_tokens_explicit"):
        return int(ctx.connection.get("max_new_tokens") or DEFAULT_ANSWER_CAP)
    if config.get("max_new_tokens"):
        return int(config["max_new_tokens"])
    return suite_spec(suite).answer_cap


def _prompt_budget(ctx, config: dict, suite: str) -> int:
    if config.get("prompt_budget"):
        return int(config["prompt_budget"])
    if ctx.connection.get("prompt_budget"):
        return int(ctx.connection["prompt_budget"])
    return suite_spec(suite).prompt_budget


def _hit_token_cap(result, cap: int) -> bool:
    meta = dict(getattr(result, "metadata", None) or {})
    if "hit_token_cap" in meta:
        return bool(meta.get("hit_token_cap"))
    completion = meta.get("completion_tokens")
    if completion is None:
        completion = getattr(result, "tokens_used", None)
    if completion is None:
        return False
    return int(completion) >= int(cap)


def _prepare_context(ctx, prompt_budget: int, answer_cap: int) -> int:
    think = int(ctx.connection.get("thinking_max_tokens") or 0)
    needed = derive_max_context(prompt_budget, think, answer_cap)
    model = ctx.model
    current = getattr(model, "max_context", None)
    if current is not None:
        try:
            model.max_context = max(int(current), needed)
        except (TypeError, ValueError):
            model.max_context = needed
    return needed


def _shots_for(category: str, validation: list[dict]) -> list[dict]:
    return [row for row in validation if str(row.get("category") or "") == category][:5]


def _prompt_for(suite: str, row: dict, validation: list[dict]) -> tuple[str, dict]:
    """Return the model prompt and private scoring context.

    For GPQA the prompt contains the question. The caller must not persist it.
    """
    if suite == "mmlu_pro":
        shots = _shots_for(str(row.get("category") or ""), validation)
        return mmlu_prompt(row, shots), {"gold": str(row.get("answer") or "")}
    if suite == "gpqa_diamond":
        options, letter = shuffle_options(
            str(row.get("id") or ""),
            str(row.get("correct") or ""),
            list(row.get("incorrect") or []),
        )
        return gpqa_prompt(str(row.get("question") or ""), options), {"gold": letter}
    if suite == "gsm8k":
        return gsm_prompt(str(row.get("question") or "")), {"gold": str(row.get("answer") or "")}
    if suite == "math500":
        return math_prompt(str(row.get("problem") or "")), {"gold": str(row.get("answer") or "")}
    if suite == "ifeval":
        return str(row.get("prompt") or ""), {"row": row}
    if suite == "humaneval_plus":
        return humaneval_prompt(row), {"row": row}
    if suite == "mbpp_plus":
        return mbpp_prompt(row), {"row": row}
    raise KeyError(suite)


def _score(suite: str, response: str, private: dict, limits: RunLimits) -> dict:
    if suite == "mmlu_pro":
        return score_mmlu_pro(private["gold"], response)
    if suite == "gpqa_diamond":
        return score_gpqa(private["gold"], response)
    if suite == "gsm8k":
        return score_gsm8k(private["gold"], response)
    if suite == "math500":
        return score_math(private["gold"], response)
    if suite == "ifeval":
        return score_ifeval_item(private["row"], response)
    if suite in {"humaneval_plus", "mbpp_plus"}:
        return score_code_task(private["row"], response, suite=suite, limits=limits)
    raise KeyError(suite)


def _row_id(suite: str, index: int, row: dict) -> str:
    for key in ("task_id", "id", "question_id", "key", "unique_id"):
        if row.get(key) not in (None, ""):
            return f"{suite}:{row.get(key)}"
    return f"{suite}:{index}"


def _aggregate(suite: str, items: list[dict]) -> dict:
    scored = [
        item
        for item in items
        if not item.get("prompt_over_budget") and not item.get("reference_excluded")
    ]
    n_scored = len(scored)
    over = sum(1 for item in items if item.get("prompt_over_budget"))

    def rate(flag: str) -> float | None:
        if not n_scored:
            return None
        return sum(1 for item in scored if item.get(flag)) / n_scored

    metrics: dict = {
        "n_items": len(items),
        "n_scored": n_scored,
        "n_over_budget": over,
        "counts_toward_dow_score": False,
    }
    if suite == "ifeval":
        summary = aggregate_ifeval(scored)
        metrics.update(summary)
        metrics["score"] = summary.get("prompt_strict")
    elif suite in {"humaneval_plus", "mbpp_plus"}:
        metrics["pass_at_1_base"] = rate("base_pass")
        metrics["pass_at_1_plus"] = rate("plus_pass")
        metrics["score"] = metrics["pass_at_1_plus"]
    else:
        metrics["accuracy"] = rate("passed")
        metrics["score"] = metrics["accuracy"]
    return metrics


def _not_run(suite: str, config: dict, reason: str) -> dict:
    fields = _protocol_fields(config)
    metrics = {
        "n_items": 0,
        "n_scored": 0,
        "status": "not_run",
        "counts_toward_dow_score": False,
        "full_size": suite_spec(suite).full_size,
        **fields,
    }
    return {
        "name": suite,
        "source": "not_run",
        "label": "Not run",
        "notes": reason,
        "items": [],
        "done": 0,
        "total": 0,
        "not_run": True,
        "status": "not_run",
        "suite_version": SUITE_VERSION,
        "metrics": metrics,
        "counts_toward_dow_score": False,
        "pass_rate": None,
        **fields,
    }


def _load_suite_rows(dest, suite: str):
    """Load rows. A missing GPQA add-on is not a failed run."""
    try:
        return load_suite(dest, suite), None
    except ManifestError as exc:
        if suite == GPQA_SUITE:
            return (None, {}), str(exc)
        raise
    except HashMismatch as exc:
        message = str(exc)
        if suite == GPQA_SUITE and "file is missing" in message:
            return (None, {}), message
        raise


def run_standard_suite(ctx, config: dict, suite: str) -> dict:
    activate_run_offline(preset={"offline": True})
    spec = suite_spec(suite)
    dest = resolve_dest(config)
    from standard_bench.bundle import verify_kit_near

    verify_kit_near(dest)
    loaded, missing = _load_suite_rows(dest, suite)
    if missing is not None:
        return _not_run(
            suite,
            config,
            "GPQA Diamond was not run. The customer kit does not include it.",
        )
    rows, entry = loaded
    dropped = exclusions_for(suite)
    if dropped:
        rows = [row for row in rows if str(row.get("task_id")) not in dropped]
    validation = load_file(dest, suite, "validation.jsonl") if suite == "mmlu_pro" else []
    limit = resolve_limit(config, suite)
    seed = resolve_seed(config)
    rows = select_rows(rows, limit, seed)
    answer_cap = _answer_cap(ctx, config, suite)
    prompt_budget = _prompt_budget(ctx, config, suite)
    fields = _protocol_fields(config)
    model_name = ctx.connection.get("model") or ctx.connection.get("name")
    applied_think, configured_think = resolve_thinking_budget(model_name, config)
    ctx.connection["thinking_max_tokens"] = applied_think
    _prepare_context(ctx, prompt_budget, answer_cap)
    code_limits = RunLimits(
        timeout_s=float(config.get("sandbox_timeout") or DEFAULT_TIMEOUT_S),
        memory_mb=int(config.get("sandbox_memory_mb") or DEFAULT_MEMORY_MB),
        cpu_s=int(config.get("sandbox_cpu_s") or DEFAULT_CPU_S),
    )
    items = []
    done = 0
    total = len(rows)
    redact = suite == "gpqa_diamond"
    for index, row in enumerate(rows):
        item_id = _row_id(suite, index, row)
        if item_id in ctx.completed_ids:
            done += 1
            continue
        if _cancelled(ctx):
            return _partial(suite, items, total, done)
        prompt, private = _prompt_for(suite, row, validation)
        result = ctx.model.generate(
            prompt,
            max_tokens=int(answer_cap),
            thinking_max_tokens=ctx.connection.get("thinking_max_tokens"),
            prompt_budget=int(prompt_budget),
        )
        meta = dict(getattr(result, "metadata", None) or {})
        raw_response = meta.get("raw_text")
        if raw_response is None:
            raw_response = result.text
        answer = strip_think_blocks(result.text, strip_edges=suite not in {"humaneval_plus", "mbpp_plus"})
        over_budget = bool(meta.get("prompt_over_budget"))
        if over_budget:
            scored = {
                "predicted": "",
                "expected": "" if redact else str(private.get("gold") or ""),
                "passed": False,
                "score": None,
                "metric": "prompt_over_budget",
            }
        else:
            scored = _score(suite, answer, private, code_limits)
        stored_prompt = "" if redact else prompt
        stored_response = "" if redact else answer
        stored_raw = "" if redact else raw_response
        item = {
            "id": item_id,
            "suite": suite,
            "category": "public_benchmark",
            "counts_toward_score": False,
            "counts_toward_dow_score": COUNTS_TOWARD_DOW_SCORE,
            "type": spec.name,
            "prompt": stored_prompt,
            "response": stored_response,
            "raw_response": stored_raw,
            "redacted": redact,
            "empty": not bool(str(answer or "").strip()),
            "expected": "" if redact else scored.get("expected", ""),
            "predicted": scored.get("predicted", ""),
            "hit_token_cap": False if over_budget else _hit_token_cap(result, answer_cap),
            "think_truncated": bool(meta.get("think_truncated")),
            "stopped_on_newline_run": bool(meta.get("stopped_on_newline_run")),
            "prompt_over_budget": over_budget,
            "prompt_budget_error": "" if redact else (meta.get("prompt_budget_error") or ""),
            "source": "live",
            "suite_version": SUITE_VERSION,
            "dataset_revision": entry.get("revision") or "",
            "passed": scored.get("passed"),
            "score": scored.get("score"),
            "base_pass": scored.get("base_pass"),
            "plus_pass": scored.get("plus_pass"),
            "prompt_strict": scored.get("prompt_strict"),
            "prompt_loose": scored.get("prompt_loose"),
            "instruction_strict": scored.get("instruction_strict"),
            "instruction_loose": scored.get("instruction_loose"),
            "sandbox_isolation": scored.get("isolation") or "",
            "preset": fields["preset"],
            "subset": fields["subset"],
            "protocol": fields["protocol"],
            "seed": fields["seed"],
            "subset_limit": limit,
            "answer_cap": answer_cap,
            "prompt_budget": prompt_budget,
            "thinking_budget": applied_think,
            "thinking_cap": configured_think,
        }
        if redact:
            # The question must not survive in any string field.
            for key, value in list(item.items()):
                if isinstance(value, str) and value and key not in {"id", "suite", "predicted", "expected"}:
                    if str(row.get("question") or "") and str(row.get("question")) in value:
                        item[key] = ""
        items.append(item)
        if ctx.on_item is not None:
            ctx.on_item(item)
        done += 1
        if ctx.on_progress is not None:
            ctx.on_progress(done)
    metrics = _aggregate(suite, items)
    metrics["full_size"] = spec.full_size
    metrics["limit"] = limit
    metrics["prompt_budget"] = prompt_budget
    metrics["answer_cap"] = answer_cap
    metrics["thinking_budget"] = applied_think
    metrics["thinking_cap"] = configured_think
    metrics["dataset_revision"] = entry.get("revision") or ""
    metrics.update(fields)
    if suite in {"humaneval_plus", "mbpp_plus"}:
        metrics["sandbox"] = describe_isolation()
        metrics["sandbox_timeout_s"] = code_limits.timeout_s
        metrics["sandbox_memory_mb"] = code_limits.memory_mb
        metrics["sandbox_cpu_s"] = code_limits.cpu_s
        metrics["reference_exclusions"] = [
            {"task_id": task_id, "reason": reason} for task_id, reason in sorted(dropped.items())
        ]
    notes = (
        f"{LABELS[suite]} ({SUITE_VERSION}). "
        f"Full size {spec.full_size}. "
        "This score is not part of the AIEval DoW Bench Score. "
        f"Protocol {fields['protocol']}, subset {fields['subset']}, "
        f"{len(items)} items, answer cap {answer_cap}, "
        f"thinking budget {applied_think}, thinking cap {configured_think}."
    )
    if limit is not None:
        notes += f" Limit {limit}."
    if dropped:
        notes += " Reference exclusions: " + ", ".join(sorted(dropped)) + "."
    return {
        "name": suite,
        "source": "live",
        "label": LABELS[suite],
        "notes": notes,
        "items": items,
        "done": done,
        "total": total,
        "suite_version": SUITE_VERSION,
        "metrics": metrics,
        "counts_toward_dow_score": False,
        "pass_rate": metrics.get("score"),
        **fields,
    }


def _partial(suite: str, items: list[dict], total: int, done: int) -> dict:
    return {
        "name": suite,
        "source": "live",
        "label": LABELS[suite],
        "notes": "Stopped because the run was cancelled.",
        "items": items,
        "done": done,
        "total": total,
        "cancelled": True,
        "suite_version": SUITE_VERSION,
        "metrics": _aggregate(suite, items),
        "counts_toward_dow_score": False,
    }


def planned_count(suite: str, config: dict) -> int:
    spec = suite_spec(suite)
    limit = resolve_limit(config or {}, suite)
    if limit is None:
        return spec.full_size
    return min(spec.full_size, int(limit))
