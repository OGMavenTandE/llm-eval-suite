"""Fetch, run, and export the public capability suites."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dow_bench.meta import derive_max_context, thinking_cap

from standard_bench import SUITE_NAMES
from standard_bench.catalog import SUITES
from standard_bench.export import write_export
from standard_bench.fetch import FetchError, fetch
from standard_bench.manifest import HashMismatch, ManifestError
from standard_bench.runner import parse_limits

DEFAULT_PROMPT_BUDGET = 4096


def _dataset_path(path: str | None, dest: Path) -> str:
    if path:
        return path
    marker = dest / "_run_marker.jsonl"
    if not marker.is_file():
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(
            json.dumps({"prompt": "Reply with the single word pong.", "expected_answer": "pong"}) + "\n",
            encoding="utf-8",
        )
    return str(marker)


def build_connection(args: argparse.Namespace) -> dict:
    answer_cap = int(args.max_new_tokens) if args.max_new_tokens else 1024
    prompt_budget = int(args.prompt_budget)
    thinking_budget = thinking_cap(args.model)
    if args.max_context:
        max_context = int(args.max_context)
    else:
        max_context = derive_max_context(prompt_budget, thinking_budget, answer_cap)
    return {
        "type": "hf",
        "name": args.model,
        "model": args.model,
        "folder": args.folder or "stub",
        "precision": args.precision,
        "mode": "chat",
        "max_new_tokens": answer_cap,
        "max_new_tokens_explicit": args.max_new_tokens is not None,
        "max_context": max_context,
        "prompt_budget": prompt_budget,
        "thinking_max_tokens": thinking_budget,
        "trust_remote_code": False,
    }


def _apply_run_env(args: argparse.Namespace) -> None:
    import os

    if args.fixtures:
        os.environ["LLM_EVAL_STANDARD_FIXTURES"] = "1"
        os.environ.pop("LLM_EVAL_STANDARD_DATA", None)
    elif args.data:
        os.environ["LLM_EVAL_STANDARD_DATA"] = args.data
        os.environ.pop("LLM_EVAL_STANDARD_FIXTURES", None)
    if args.limit is not None:
        os.environ["LLM_EVAL_STANDARD_LIMIT"] = str(args.limit)
    else:
        os.environ.pop("LLM_EVAL_STANDARD_LIMIT", None)
    if args.limits:
        os.environ["LLM_EVAL_STANDARD_LIMITS"] = args.limits
    else:
        os.environ.pop("LLM_EVAL_STANDARD_LIMITS", None)
    if args.seed is not None:
        os.environ["LLM_EVAL_STANDARD_SEED"] = str(args.seed)


def _run(args: argparse.Namespace) -> int:
    from llm_eval.offline import activate_run_offline
    from llm_eval_suite.runs import RunManager

    from standard_bench.stub import StubStandardModel

    activate_run_offline(preset={"offline": True})
    _apply_run_env(args)
    if args.fixtures:
        dest = Path(__file__).resolve().parent / "fixtures"
    else:
        if not args.data:
            print("Pass --data, the directory from fetch, or pass --fixtures.", file=sys.stderr)
            return 2
        dest = Path(args.data)
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    connection = build_connection(args)
    if args.stub or args.fixtures:
        manager = RunManager(runs_dir, model_factory=lambda _profile: StubStandardModel(args.model))
    else:
        manager = RunManager(runs_dir)
    try:
        record = manager.start(
            connection=connection,
            preset_id="standard_capability",
            dataset_path=_dataset_path(args.dataset, runs_dir),
            background=False,
        )
    except (HashMismatch, ManifestError, FetchError) as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if record.get("status") == "failed":
        message = str(record.get("error") or "run failed")
        print(message, file=sys.stderr)
        lowered = message.lower()
        if "hash mismatch" in lowered or "manifest" in lowered:
            return 2
        return 1
    paths = write_export(runs_dir, output)
    print(paths["csv"])
    print(paths["json"])
    return 0 if record.get("status") in {"completed", "invalid"} else 1


def _fetch(args: argparse.Namespace) -> int:
    names = [part.strip() for part in (args.suites or "").split(",") if part.strip()] or None
    try:
        fetch(args.dest, names, allow_moved=bool(args.allow_moved))
    except FetchError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    return 0


def _export(args: argparse.Namespace) -> int:
    paths = write_export(args.runs_dir, args.output)
    print(paths["csv"])
    print(paths["json"])
    return 0


def _smoke(args: argparse.Namespace) -> int:
    args.fixtures = True
    args.stub = True
    args.data = ""
    args.limit = None
    args.limits = ""
    args.seed = None
    if not args.output:
        args.output = str(Path(args.runs_dir) / "public_benchmarks.csv")
    return _run(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-eval-standard",
        description="Public capability benchmarks. Scores stay out of the DoW bench composite.",
    )
    sub = parser.add_subparsers(dest="command")

    fetched = sub.add_parser("fetch", help="Download pinned datasets on a connected machine")
    fetched.add_argument("--dest", required=True)
    fetched.add_argument("--suites", default="", help="Comma-separated suite names. Default: all.")
    fetched.add_argument(
        "--allow-moved",
        action="store_true",
        help="Pin the live revision when it differs from the catalog pin.",
    )
    fetched.set_defaults(func=_fetch)

    live = sub.add_parser("run", help="Score the standard_capability preset")
    live.add_argument("--data", default="")
    live.add_argument("--fixtures", action="store_true", help="Use the packaged synthetic fixtures")
    live.add_argument("--stub", action="store_true")
    live.add_argument("--output", required=True)
    live.add_argument("--runs-dir", default="runs")
    live.add_argument("--model", default="OLMo 2 1B Instruct")
    live.add_argument("--folder", default="")
    live.add_argument("--precision", default="fp16")
    live.add_argument("--max-new-tokens", type=int, default=None)
    live.add_argument("--max-context", type=int, default=None)
    live.add_argument("--prompt-budget", type=int, default=DEFAULT_PROMPT_BUDGET)
    live.add_argument("--dataset", default="")
    live.add_argument("--limit", type=int, default=None, help="First N items of each suite")
    live.add_argument("--limits", default="", help="Per-suite caps, for example mmlu_pro=200,gsm8k=200")
    live.add_argument("--seed", type=int, default=None, help="Shuffle before --limit")
    live.set_defaults(func=_run)

    exported = sub.add_parser("export", help="Write public benchmark CSV and JSON from finished runs")
    exported.add_argument("--runs-dir", required=True)
    exported.add_argument("--output", required=True)
    exported.set_defaults(func=_export)

    smoke = sub.add_parser("smoke", help="Stub model on the synthetic fixtures")
    smoke.add_argument("--output", default="")
    smoke.add_argument("--runs-dir", default="runs")
    smoke.add_argument("--model", default="OLMo 2 1B Instruct")
    smoke.add_argument("--folder", default="")
    smoke.add_argument("--precision", default="fp16")
    smoke.add_argument("--max-new-tokens", type=int, default=None)
    smoke.add_argument("--max-context", type=int, default=None)
    smoke.add_argument("--prompt-budget", type=int, default=DEFAULT_PROMPT_BUDGET)
    smoke.add_argument("--dataset", default="")
    smoke.set_defaults(func=_smoke)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        raise SystemExit(2)
    if getattr(args, "limits", "") and getattr(args, "command", "") == "run":
        parse_limits(args.limits)
    raise SystemExit(args.func(args))


# Re-export for tests that inspect the suite list.
KNOWN_SUITES = SUITE_NAMES
CATALOG = SUITES
