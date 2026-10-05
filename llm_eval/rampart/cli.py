"""CLI for the offline RAMPART pytest smoke crib."""

import argparse

from llm_eval.rampart.smoke import DEFAULT_OUTPUT, run_smoke


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-eval-rampart",
        description=(
            "Write a RAMPART-shaped pytest smoke report for llm-eval-suite. "
            "Does not call an LLM and does not import rampart."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser(
        "smoke",
        help="Replay the upstream LLM-free smoke fixture. No model and no rampart install.",
    )
    smoke.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output JSON path (default: {DEFAULT_OUTPUT}).",
    )
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "smoke":
        report, path = run_smoke(args.output)
        print(
            f"Wrote RAMPART-shaped smoke report to {path}\n"
            f"live_llm=false cases={len(report.cases)} source={report.source}"
        )
        return

    parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
