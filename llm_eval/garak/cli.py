"""CLI for the offline garak fixture eval crib."""

import argparse
import sys

from llm_eval.garak.smoke import DEFAULT_FIXTURE, DEFAULT_OUTPUT, run_smoke


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-eval-garak",
        description=(
            "Write a garak-shaped JSONL report for llm-eval-suite. "
            "Does not call a model and does not import garak."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser(
        "smoke",
        help="Score the canned probe/response fixture. No model and no garak install.",
    )
    smoke.add_argument(
        "--fixture",
        default=str(DEFAULT_FIXTURE),
        help=f"Canned probe JSONL (default: {DEFAULT_FIXTURE}).",
    )
    smoke.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT}).",
    )
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "smoke":
        try:
            report, path = run_smoke(args.output, args.fixture)
        except (OSError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        print(
            f"Wrote garak-shaped smoke report to {path}\n"
            f"live_llm=false live_api=false attempts={len(report.attempts)} "
            f"probes={len(report.summaries)} source={report.setup.source}"
        )
        return

    parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
