"""Command line for the dry run, export, judge pass, sample, and agreement."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dow_bench.export import write_export
from dow_bench.judge import DEFAULT_JUDGE, grade_run_dir, load_family_map
from dow_bench.meta import exclusion_reason
from dow_bench.sample import agreement, load_items_for_agreement, write_sheet
from dow_bench.stub import StubDowModel, stub_judge_reply


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


def dry_run(args: argparse.Namespace) -> int:
    from llm_eval_suite.runs import RunManager

    if exclusion_reason(args.model):
        print(exclusion_reason(args.model), file=sys.stderr)
        return 2
    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    connection = {
        "type": "hf",
        "name": args.model,
        "model": args.model,
        "folder": args.folder or "stub",
        "precision": args.precision,
        "mode": "chat",
        "max_new_tokens": args.max_new_tokens,
        "trust_remote_code": False,
    }
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
    if args.stub:
        generate = stub_judge_reply
        judge_name = args.judge_model
    else:
        from llm_eval_suite.connections import build_model

        profile = {
            "type": "hf",
            "model": args.judge_model,
            "folder": args.judge_folder,
            "precision": args.precision,
            "trust_remote_code": bool(args.trust_remote_code),
            "max_new_tokens": args.max_new_tokens,
            "mode": "chat",
        }
        model = build_model(profile)
        judge_name = args.judge_model

        def generate(prompt: str) -> str:
            return model.generate(prompt, max_tokens=args.max_new_tokens).text

    result = grade_run_dir(
        args.run_dir,
        judge_name=judge_name,
        judge_generate=generate,
        family_map=load_family_map(args.families) if args.families else load_family_map(),
        model_name=args.model,
    )
    print(json.dumps(result))
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-eval-dow", description="Department of War bench tools")
    parser.add_argument("--stub", action="store_true", help="Dry-run dow_bench with the stub model")
    parser.add_argument("--output", default="")
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--model", default="OLMo 2 1B Instruct")
    parser.add_argument("--precision", default="fp16")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--folder", default="")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE)
    sub = parser.add_subparsers(dest="command")

    dry = sub.add_parser("dry-run", help="Run dow_bench and write the CSV")
    dry.add_argument("--stub", action="store_true")
    dry.add_argument("--output", required=True)
    dry.add_argument("--runs-dir", default="runs")
    dry.add_argument("--model", default="OLMo 2 1B Instruct")
    dry.add_argument("--precision", default="fp16")
    dry.add_argument("--max-new-tokens", type=int, default=256)
    dry.add_argument("--dataset", default="")
    dry.add_argument("--folder", default="")
    dry.add_argument("--judge-model", default=DEFAULT_JUDGE)
    dry.set_defaults(func=dry_run)

    exported = sub.add_parser("export", help="Write the leaderboard CSV from finished runs")
    exported.add_argument("--runs-dir", required=True)
    exported.add_argument("--output", required=True)
    exported.set_defaults(func=export_only)

    judge = sub.add_parser("judge", help="Judge-only pass over a finished run")
    judge.add_argument("--run-dir", required=True)
    judge.add_argument("--stub", action="store_true")
    judge.add_argument("--judge-model", default=DEFAULT_JUDGE)
    judge.add_argument("--judge-folder", default="")
    judge.add_argument("--model", default="")
    judge.add_argument("--precision", default="fp16")
    judge.add_argument("--max-new-tokens", type=int, default=256)
    judge.add_argument("--trust-remote-code", action="store_true")
    judge.add_argument("--families", default="")
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
