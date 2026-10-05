"""CLI for the offline Dioptra light pilot."""

import argparse
import sys

from llm_eval.dioptra.export import export_audit_file, write_bundle
from llm_eval.dioptra.smoke import DEFAULT_OUTPUT, run_smoke


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-eval-dioptra",
        description=(
            "Write Dioptra-shaped experiment JSON for llm-eval-suite. "
            "Does not contact a Dioptra server."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    smoke = subparsers.add_parser(
        "smoke",
        help="Write a one-job public fixture. No model and no Dioptra server.",
    )
    smoke.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output JSON path (default: {DEFAULT_OUTPUT}).",
    )

    export = subparsers.add_parser(
        "export-audit",
        help="Map an existing suite audit JSON file to a Dioptra-shaped record.",
    )
    export.add_argument("audit", help="Path to an audit JSON file.")
    export.add_argument("--output", required=True, help="Output JSON path.")
    return parser


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "smoke":
        bundle, path = run_smoke(args.output)
        job_count = len(bundle.experiment.jobs)
        metric_count = sum(len(job.metrics) for job in bundle.experiment.jobs)
        print(
            f"Wrote Dioptra-shaped experiment to {path}\n"
            f"live_api=false jobs={job_count} metrics={metric_count} source={bundle.source}"
        )
        return

    if args.command == "export-audit":
        try:
            bundle = export_audit_file(args.audit)
        except FileNotFoundError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        except (OSError, ValueError) as exc:
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(1)
        path = write_bundle(bundle, args.output)
        print(f"Wrote Dioptra-shaped experiment to {path}")
        for warning in bundle.warnings:
            print(f"Warning: {warning}", file=sys.stderr)
        return

    parser.error(f"unknown command: {args.command}")


if __name__ == "__main__":
    main()
