"""Command line for the run, dry run, export, rescore, judge pass, sample, and agreement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dow_bench.export import write_export
from dow_bench.judge import (
    DEFAULT_JUDGE,
    DEFAULT_JUDGE_MAX_CONTEXT,
    grade_run_dir,
    load_family_map,
    resolve_model_name,
    same_family,
)
from dow_bench.meta import DEFAULT_PROMPT_BUDGET, derive_max_context, exclusion_reason, thinking_cap
from dow_bench.rescore import rescore_run_dir
from dow_bench.sample import agreement, load_items_for_agreement, write_sheet
from dow_bench.stub import StubDowModel, stub_judge_reply

_MAX_NEW_TOKENS_HELP = (
    "Answer cap for each item. Default 1024. "
    "Thinking models use a separate thinking budget from the model list. "
    "An item whose answer phase stops on this cap is stored with hit_token_cap true and counted in run.json."
)
_MAX_CONTEXT_HELP = (
    "Context window for the answer call. "
    "The default is the prompt budget plus the thinking budget plus the answer cap. "
    "The default prompt budget is 512. The longest stage-1 prompt is 216 tokens with the chat template."
)
_PROMPT_BUDGET_HELP = (
    "Tokens reserved for the prompt. Default 512. "
    "A longer prompt is stored as prompt_over_budget and is not truncated."
)


def _dataset_path(path: str | None) -> str:
    if path:
        return path
    sample = Path(__file__).resolve().parents[1] / "datasets" / "sample_factcheck_50.jsonl"
    if sample.is_file():
        return str(sample)
    fallback = Path("dow-bench-dataset.jsonl")
    if not fallback.is_file():
        fallback.write_text(
            json.dumps({"prompt": "Reply with the single word pong.", "expected_answer": "pong"}) + "\n",
            encoding="utf-8",
        )
    return str(fallback)


def run_limits(args: argparse.Namespace) -> dict:
    """Prompt budget, thinking budget, answer cap, and the context window they need."""
    answer_cap = int(args.max_new_tokens)
    prompt_budget = int(getattr(args, "prompt_budget", None) or DEFAULT_PROMPT_BUDGET)
    thinking_budget = thinking_cap(getattr(args, "model", None))
    explicit = getattr(args, "max_context", None)
    if explicit:
        max_context = int(explicit)
    else:
        max_context = derive_max_context(prompt_budget, thinking_budget, answer_cap)
    return {
        "prompt_budget": prompt_budget,
        "thinking_budget": thinking_budget,
        "answer_cap": answer_cap,
        "max_context": max_context,
    }


def build_run_connection(args: argparse.Namespace) -> dict:
    limits = run_limits(args)
    return {
        "type": "hf",
        "name": args.model,
        "model": args.model,
        "folder": args.folder or "stub",
        "precision": args.precision,
        "mode": "chat",
        "max_new_tokens": limits["answer_cap"],
        "max_new_tokens_explicit": True,
        "max_context": limits["max_context"],
        "prompt_budget": limits["prompt_budget"],
        "thinking_max_tokens": limits["thinking_budget"],
        "trust_remote_code": False,
    }


def judge_limits(args: argparse.Namespace, model_name: str | None = None) -> dict:
    """Judge prompt budget is ``--judge-max-context``. The window also holds the reply.

    A thinking judge keeps its own thinking budget on top of that prompt budget,
    so a prompt that passed the 2048 check is not then truncated.
    """
    answer_cap = int(args.max_new_tokens)
    prompt_budget = int(args.judge_max_context)
    name = model_name if model_name is not None else getattr(args, "judge_model", None)
    thinking_budget = thinking_cap(name)
    return {
        "prompt_budget": prompt_budget,
        "thinking_budget": thinking_budget,
        "answer_cap": answer_cap,
        "max_context": derive_max_context(prompt_budget, thinking_budget, answer_cap),
    }


def build_judge_profile(args: argparse.Namespace, model_name: str | None = None, folder: str | None = None) -> dict:
    name = model_name if model_name is not None else args.judge_model
    limits = judge_limits(args, name)
    return {
        "type": "hf",
        "model": name,
        "folder": args.judge_folder if folder is None else folder,
        "precision": args.precision,
        "trust_remote_code": bool(args.trust_remote_code),
        "max_new_tokens": limits["answer_cap"],
        "max_context": limits["max_context"],
        "prompt_budget": limits["prompt_budget"],
        "thinking_max_tokens": limits["thinking_budget"],
        "mode": "chat",
    }


def _candidate_from_run(run_dir: str, explicit: str) -> str:
    path = Path(run_dir) / "run.json"
    record = {}
    if path.is_file():
        record = json.loads(path.read_text(encoding="utf-8"))
    return resolve_model_name(explicit, record)


def dry_run(args: argparse.Namespace) -> int:
    from llm_eval_suite.runs import RunManager

    if exclusion_reason(args.model):
        print(exclusion_reason(args.model), file=sys.stderr)
        return 2
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    connection = build_run_connection(args)
    if args.stub:
        manager = RunManager(runs_dir, model_factory=lambda _profile: StubDowModel(args.model))
    else:
        manager = RunManager(runs_dir)
    record = manager.start(
        connection=connection,
        preset_id="dow_bench",
        dataset_path=_dataset_path(args.dataset),
        background=False,
    )
    run_dir = runs_dir / record["run_id"]
    if args.stub:
        grade_run_dir(
            run_dir,
            judge_name=args.judge_model,
            judge_generate=stub_judge_reply,
            family_map=load_family_map(),
            model_name=args.model,
        )
    paths = write_export(runs_dir, output)
    print(paths["csv"])
    return 0 if record.get("status") in {"completed", "invalid"} else 1


def export_only(args: argparse.Namespace) -> int:
    paths = write_export(args.runs_dir, args.output)
    print(paths["csv"])
    return 0


def judge_only(args: argparse.Namespace) -> int:
    count_tokens = None
    family_map = load_family_map(args.families) if args.families else load_family_map()
    model_name = _candidate_from_run(args.run_dir, args.model)
    fallback_name = str(getattr(args, "fallback_judge", "") or "").strip()
    fallback_generate = None
    if args.stub:
        generate = stub_judge_reply
        judge_name = args.judge_model
        if fallback_name:
            fallback_generate = stub_judge_reply
    else:
        from llm_eval_suite.connections import build_model

        profile = build_judge_profile(args)
        model = build_model(profile)
        judge_name = args.judge_model
        limits = judge_limits(args)

        def count_tokens(prompt: str) -> int:
            return model.count_prompt_tokens(prompt)

        def generate(prompt: str) -> str:
            return model.generate(
                prompt,
                max_tokens=limits["answer_cap"],
                prompt_budget=limits["prompt_budget"],
                thinking_max_tokens=limits["thinking_budget"],
            ).text

        use_fallback = bool(
            fallback_name
            and same_family(model_name, judge_name, family_map)
            and not same_family(model_name, fallback_name, family_map)
        )
        if use_fallback:
            if not str(args.fallback_judge_folder or "").strip():
                print("--fallback-judge-folder is required for a live fallback judge", file=sys.stderr)
                return 2
            fallback_profile = build_judge_profile(args, fallback_name, args.fallback_judge_folder)
            fallback_model = build_model(fallback_profile)
            fallback_limits = judge_limits(args, fallback_name)

            def fallback_generate(prompt: str, _model=fallback_model, _limits=fallback_limits) -> str:
                return _model.generate(
                    prompt,
                    max_tokens=_limits["answer_cap"],
                    prompt_budget=_limits["prompt_budget"],
                    thinking_max_tokens=_limits["thinking_budget"],
                ).text

    result = grade_run_dir(
        args.run_dir,
        judge_name=judge_name,
        judge_generate=generate,
        family_map=family_map,
        model_name=model_name,
        judge_max_context=args.judge_max_context,
        count_tokens=count_tokens,
        fallback_name=fallback_name,
        fallback_generate=fallback_generate,
    )
    print(json.dumps(result))
    return 0


def rescore_only(args: argparse.Namespace) -> int:
    try:
        result = rescore_run_dir(args.run_dir)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    print(result["csv"])
    print(result["summary"])
    return 0


def sample_only(args: argparse.Namespace) -> int:
    result = write_sheet(args.runs_dir, args.output_dir, seed=args.seed)
    print(result["markdown"])
    print(result["html"])
    return 0


def agreement_only(args: argparse.Namespace) -> int:
    items = load_items_for_agreement(args.run_dir)
    result = agreement(args.grades, items)
    print(json.dumps(result))
    return 0


def _add_run_budget_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-new-tokens", type=int, default=1024, help=_MAX_NEW_TOKENS_HELP)
    parser.add_argument("--max-context", type=int, default=None, help=_MAX_CONTEXT_HELP)
    parser.add_argument("--prompt-budget", type=int, default=DEFAULT_PROMPT_BUDGET, help=_PROMPT_BUDGET_HELP)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-eval-dow", description="Department of War bench tools")
    parser.add_argument("--stub", action="store_true", help="Dry-run dow_bench with the stub model")
    parser.add_argument("--output", default="")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--model", default="OLMo 2 1B Instruct")
    parser.add_argument("--precision", default="fp16")
    _add_run_budget_args(parser)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--folder", default="")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE)
    sub = parser.add_subparsers(dest="command")

    dry = sub.add_parser("dry-run", help="Run the Department of War bench and write the CSV")
    dry.add_argument("--stub", action="store_true")
    dry.add_argument("--output", required=True)
    dry.add_argument("--runs-dir", default="runs")
    dry.add_argument("--model", default="OLMo 2 1B Instruct")
    dry.add_argument("--precision", default="fp16")
    _add_run_budget_args(dry)
    dry.add_argument("--dataset", default="")
    dry.add_argument("--folder", default="")
    dry.add_argument("--judge-model", default=DEFAULT_JUDGE)
    dry.set_defaults(func=dry_run)

    live = sub.add_parser("run", help="Run the Department of War bench and write the CSV")
    live.add_argument("--stub", action="store_true")
    live.add_argument("--output", required=True)
    live.add_argument("--runs-dir", default="runs")
    live.add_argument("--model", default="OLMo 2 1B Instruct")
    live.add_argument("--precision", default="fp16")
    _add_run_budget_args(live)
    live.add_argument("--dataset", default="")
    live.add_argument("--folder", default="")
    live.add_argument("--judge-model", default=DEFAULT_JUDGE)
    live.set_defaults(func=dry_run)

    exported = sub.add_parser("export", help="Write the leaderboard CSV from finished runs")
    exported.add_argument("--runs-dir", required=True)
    exported.add_argument("--output", required=True)
    exported.set_defaults(func=export_only)

    rescore = sub.add_parser(
        "rescore",
        help="Recompute deterministic scores from saved responses. Does not load a model.",
    )
    rescore.add_argument("--run-dir", required=True)
    rescore.set_defaults(func=rescore_only)

    judge = sub.add_parser("judge", help="Judge-only pass over a finished run")
    judge.add_argument("--run-dir", required=True)
    judge.add_argument("--stub", action="store_true")
    judge.add_argument("--judge-model", default=DEFAULT_JUDGE)
    judge.add_argument("--judge-folder", default="")
    judge.add_argument("--model", default="")
    judge.add_argument("--precision", default="fp16")
    judge.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens for the judge reply. This is not the answer cap.",
    )
    judge.add_argument(
        "--judge-max-context",
        type=int,
        default=DEFAULT_JUDGE_MAX_CONTEXT,
        help=(
            "Maximum tokens in the judge prompt. Default 2048. "
            "The judge context window is this budget plus the judge thinking budget plus the reply cap. "
            "A longer prompt is stored as judge_status over_budget and is not truncated."
        ),
    )
    judge.add_argument("--trust-remote-code", action="store_true")
    judge.add_argument("--families", default="")
    judge.add_argument(
        "--fallback-judge",
        default="",
        help=(
            "Judge used only when the primary judge is the same family as the model under test. "
            "With no fallback, those items are judge_skipped and are not counted as fails."
        ),
    )
    judge.add_argument(
        "--fallback-judge-folder",
        default="",
        help="Local folder for --fallback-judge. Required when the fallback is a live model.",
    )
    judge.set_defaults(func=judge_only)

    sample = sub.add_parser("sample", help="Write the 30-item grading sheet")
    sample.add_argument("--runs-dir", required=True)
    sample.add_argument("--output-dir", required=True)
    sample.add_argument("--seed", type=int, default=20261008)
    sample.set_defaults(func=sample_only)

    agree = sub.add_parser("agreement", help="Raw agreement of a filled grade CSV against the judge")
    agree.add_argument("--grades", required=True)
    agree.add_argument("--run-dir", required=True)
    agree.set_defaults(func=agreement_only)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "func", None) is None:
        if args.stub or args.output:
            if not args.output:
                parser.error("--output is required for the dry run")
            code = dry_run(args)
        else:
            parser.print_help()
            code = 2
    else:
        code = args.func(args)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
