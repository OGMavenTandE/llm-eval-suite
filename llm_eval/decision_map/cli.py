"""Validate, resolve, and render a decision-to-evidence map.

Print to PDF from the browser. There is no browser dependency in this command.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

from llm_eval.decision_map.load import MapError, load_map, load_runs
from llm_eval.decision_map.render import render_html
from llm_eval.decision_map.resolve import resolve_map
from llm_eval.decision_map.status import as_utc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-eval-map",
        description=(
            "Validate a decision-to-evidence map, resolve evidence status from "
            "click-through runs, and write a one-page HTML file. "
            "The command reports evidence status only."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    validate = sub.add_parser("validate", help="Check the map file, metric paths, and wording.")
    validate.add_argument("map", help="Path to the map YAML.")

    resolve = sub.add_parser("resolve", help="Write evidence status as JSON.")
    _run_args(resolve)
    resolve.add_argument("--output", help="JSON path. Omit to print JSON.")

    render = sub.add_parser(
        "render",
        help="Write self-contained HTML. Use the browser print dialog for PDF.",
    )
    _run_args(render)
    render.add_argument("--output", required=True, help="HTML path.")
    return parser


def _run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("map", help="Path to the map YAML.")
    parser.add_argument("--runs", required=True, help="A run directory, a run.json file, or a directory of runs.")
    parser.add_argument(
        "--as-of",
        help="UTC date or timestamp used for age and attestation expiry. Default is now.",
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            decision_map = load_map(args.map)
            print(f"OK {len(decision_map.decisions)} decisions in {args.map}")
            return
        if args.command == "resolve":
            payload = _resolved_json(args)
            if args.output:
                destination = Path(args.output)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(payload, encoding="utf-8")
                print(f"Wrote {destination}")
            else:
                print(payload, end="")
            return
        if args.command == "render":
            decision_map = load_map(args.map)
            resolved = resolve_map(decision_map, load_runs(args.runs), now=_as_of(args.as_of))
            html = render_html(decision_map, resolved)
            destination = Path(args.output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(html, encoding="utf-8")
            print(f"Wrote {destination}")
            print("Print to PDF from the browser: landscape, 11 by 8.5 inches, margins none, background graphics on.")
            return
    except MapError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
    parser.error(f"unknown command: {args.command}")


def _resolved_json(args) -> str:
    decision_map = load_map(args.map)
    resolved = resolve_map(decision_map, load_runs(args.runs), now=_as_of(args.as_of))
    return json.dumps(resolved.model_dump(mode="json"), indent=2) + "\n"


def _as_of(text: str | None) -> datetime | None:
    if not text:
        return None
    stripped = text.strip()
    if len(stripped) == 10:
        try:
            parsed = datetime.fromisoformat(stripped)
        except ValueError as exc:
            raise MapError([f"could not read --as-of {text!r}"]) from exc
        return as_utc(parsed)
    if stripped.endswith("Z"):
        stripped = stripped[:-1] + "+00:00"
    try:
        return as_utc(datetime.fromisoformat(stripped))
    except ValueError as exc:
        raise MapError([f"could not read --as-of {text!r}"]) from exc


if __name__ == "__main__":
    main()
