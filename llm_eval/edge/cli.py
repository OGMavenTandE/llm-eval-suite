"""Quantization comparison for one model on a local Ollama or OpenAI-compatible server."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from llm_eval.edge.quant import build_delta_report, write_reports


def parse_variant(text: str) -> tuple[str, str]:
    """``label=model``. The model may contain colons, as in ``q4=llama3.2:3b-instruct-q4_K_M``."""
    if "=" not in text:
        raise ValueError(f"Expected label=model, got {text!r}.")
    label, model = text.split("=", 1)
    label = label.strip()
    model = model.strip()
    if not label or not model:
        raise ValueError(f"Expected label=model, got {text!r}.")
    return label, model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="llm-eval-edge", description="Edge evaluation helpers.")
    sub = parser.add_subparsers(dest="command", required=True)
    quant = sub.add_parser("quant-compare", help="Run one suite on each quantization and write a delta report.")
    quant.add_argument("--baseline", required=True, help="Baseline as label=model, for example fp16=llama3.2:3b-instruct-fp16.")
    quant.add_argument("--variant", action="append", default=[], help="Another variant as label=model. Repeat for each quantization.")
    quant.add_argument("--provider", choices=("ollama", "openai"), default="ollama")
    quant.add_argument("--base-url", default="http://127.0.0.1:11434")
    quant.add_argument("--preset", default="quant_quick")
    quant.add_argument("--dataset", default="datasets/sample_factcheck_50.jsonl")
    quant.add_argument("--tolerance", type=float, default=0.05, help="Largest accuracy drop that still passes.")
    quant.add_argument("--concurrency", type=int, default=1, help="Optional load-probe concurrency recorded by the edge metrics.")
    quant.add_argument("--out", default="runs/quant-compare")
    quant.add_argument("--dry-run", action="store_true", help="Write the plan only. No model calls.")
    return parser


def run_quant_compare(spec: dict, *, model_factory=None) -> dict:
    out = Path(spec["out"])
    ordered = [spec["baseline_pair"], *spec.get("variants", [])]
    seen: set[str] = set()
    pairs = []
    for label, model in ordered:
        if label in seen:
            continue
        seen.add(label)
        pairs.append((label, model))
    if spec.get("dry_run"):
        plan = {
            "dry_run": True,
            "preset": spec["preset"],
            "provider": spec["provider"],
            "base_url": spec["base_url"],
            "dataset": spec["dataset"],
            "baseline": pairs[0][0],
            "tolerance": spec["tolerance"],
            "concurrency": spec["concurrency"],
            "variants": [{"label": label, "model": model} for label, model in pairs],
            "note": "Dry run. No model was called.",
        }
        out.mkdir(parents=True, exist_ok=True)
        (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        plan["plan"] = str(out / "plan.json")
        return plan

    from llm_eval_suite.runs import RunManager

    manager = RunManager(out / "runs", model_factory=model_factory)
    collected = []
    for label, model in pairs:
        connection = {
            "name": label,
            "type": spec["provider"],
            "model": model,
            "base_url": spec["base_url"],
            "api_key": "",
            "mode": "chat",
            "edge_concurrency": spec["concurrency"],
        }
        record = manager.start(
            connection=connection,
            preset_id=spec["preset"],
            dataset_path=spec["dataset"],
            background=False,
            in_process=True,
        )
        collected.append(
            {
                "label": label,
                "model": model,
                "run": record,
                "items": manager.items(record["run_id"]),
            }
        )
    report = build_delta_report(collected, baseline=pairs[0][0], tolerance=float(spec["tolerance"]))
    report["files"] = {
        "json": str(out / "report.json"),
        "markdown": str(out / "report.md"),
        "html": str(out / "report.html"),
    }
    write_reports(report, out)
    return report


def spec_from_args(args: argparse.Namespace) -> dict:
    baseline = parse_variant(args.baseline)
    variants = [parse_variant(item) for item in args.variant]
    if not variants:
        raise ValueError("Pass at least one --variant label=model.")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1.")
    if args.tolerance < 0:
        raise ValueError("--tolerance must be zero or positive.")
    return {
        "baseline_pair": baseline,
        "variants": variants,
        "provider": args.provider,
        "base_url": args.base_url,
        "preset": args.preset,
        "dataset": args.dataset,
        "tolerance": args.tolerance,
        "concurrency": args.concurrency,
        "out": args.out,
        "dry_run": bool(args.dry_run),
    }


def main(argv=None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command != "quant-compare":
        parser.error(f"Unknown command {args.command}.")
    try:
        spec = spec_from_args(args)
        result = run_quant_compare(spec)
    except (ValueError, KeyError, FileNotFoundError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    if result.get("dry_run"):
        print(f"Dry run plan: {result.get('plan')}")
        print("No model was called.")
        sys.exit(0)
    print(f"Verdict: {result.get('verdict')}")
    files = result.get("files") or {}
    for kind in ("json", "markdown", "html"):
        if files.get(kind):
            print(f"{kind}: {files[kind]}")
    sys.exit(0)
