"""Quantization comparison for one model on a local Ollama or OpenAI-compatible server."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from llm_eval.edge.quant import build_delta_report, write_reports
from llm_eval.edge.vram import prepare_variant_vram


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
    quant.add_argument("--provider", choices=("ollama", "openai", "vllm", "llama_cpp"), default="ollama")
    quant.add_argument("--base-url", default="http://127.0.0.1:11434")
    quant.add_argument(
        "--preset",
        default="quant_quick",
        help="Any suite preset. quant_quick is the five-question smoke. standard_capability_quick is the quick50 standard subset.",
    )
    quant.add_argument("--dataset", default="datasets/sample_factcheck_50.jsonl")
    quant.add_argument(
        "--data",
        default=None,
        help="Standard benchmark directory. Same as LLM_EVAL_STANDARD_DATA. Required for a standard preset unless that variable is already set.",
    )
    quant.add_argument(
        "--per-suite",
        type=int,
        default=None,
        help="Cap each suite at N items. Use 20 with standard_capability_quick for a shorter standard subset. A larger N does not raise quant_quick above five.",
    )
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
            "per_suite": spec.get("per_suite"),
            "data": spec.get("data") or "",
            "variants": [{"label": label, "model": model} for label, model in pairs],
            "note": "Dry run. No model was called.",
        }
        out.mkdir(parents=True, exist_ok=True)
        (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        plan["plan"] = str(out / "plan.json")
        return plan

    from llm_eval_suite.runs import RunManager

    data_path = require_standard_data(spec["preset"], spec.get("data"))
    previous_data = os.environ.get("LLM_EVAL_STANDARD_DATA")
    overwrite_data = bool(spec.get("data") and data_path)
    if overwrite_data:
        os.environ["LLM_EVAL_STANDARD_DATA"] = data_path
    manager = RunManager(out / "runs", model_factory=model_factory)
    collected = []
    try:
        for label, model in pairs:
            policy = prepare_variant_vram(spec["provider"], spec["base_url"])
            connection = {
                "name": label,
                "type": spec["provider"],
                "model": model,
                "base_url": spec["base_url"],
                "api_key": "",
                "mode": "chat",
                "edge_concurrency": spec["concurrency"],
                "edge_vram": policy,
            }
            record = manager.start(
                connection=connection,
                preset_id=spec["preset"],
                dataset_path=spec["dataset"],
                background=False,
                in_process=True,
                per_suite=spec.get("per_suite"),
            )
            collected.append(
                {
                    "label": label,
                    "model": model,
                    "run": record,
                    "items": manager.items(record["run_id"]),
                }
            )
    finally:
        if overwrite_data:
            if previous_data is None:
                os.environ.pop("LLM_EVAL_STANDARD_DATA", None)
            else:
                os.environ["LLM_EVAL_STANDARD_DATA"] = previous_data
    report = build_delta_report(collected, baseline=pairs[0][0], tolerance=float(spec["tolerance"]))
    report = attach_execution_failures(report, collected)
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
    if args.per_suite is not None and args.per_suite < 1:
        raise ValueError("--per-suite must be at least 1.")
    return {
        "baseline_pair": baseline,
        "variants": variants,
        "provider": args.provider,
        "base_url": args.base_url,
        "preset": args.preset,
        "dataset": args.dataset,
        "tolerance": args.tolerance,
        "concurrency": args.concurrency,
        "per_suite": args.per_suite,
        "data": args.data,
        "out": args.out,
        "dry_run": bool(args.dry_run),
    }


def standard_suites_for_preset(preset_id: str) -> list[str]:
    from llm_eval_suite.presets import STANDARD_SUITE_NAMES, expand_preset

    expanded = expand_preset(preset_id)
    names = []
    for row in expanded.get("suites") or []:
        name = row.get("name")
        if name in STANDARD_SUITE_NAMES:
            names.append(name)
    return names


def require_standard_data(preset_id: str, data: str | None) -> str | None:
    """Return the benchmark directory, or None when this preset does not need one.

    A standard preset with no directory raises before any model call. The
    message matches the runner. Hashes are checked for every suite except
    GPQA, which stays a recorded skip when its file is absent.
    """
    suites = standard_suites_for_preset(preset_id)
    if not suites:
        return None
    from standard_bench.manifest import HashMismatch, ManifestError, verify_suite
    from standard_bench.runner import resolve_dest

    config = {"data_dir": data} if data else {}
    try:
        dest = resolve_dest(config)
    except ManifestError as exc:
        raise ValueError(str(exc)) from exc
    if not dest.is_dir():
        raise ValueError(
            f"Standard benchmark directory does not exist: {dest}. "
            "Set LLM_EVAL_STANDARD_DATA or pass --data."
        )
    for suite in suites:
        if suite == "gpqa_diamond":
            continue
        try:
            verify_suite(dest, suite)
        except (ManifestError, HashMismatch) as exc:
            raise ValueError(str(exc)) from exc
    return str(dest)


def attach_execution_failures(report: dict, collected: list[dict]) -> dict:
    """A run that did not finish is a failed comparison, not an unscored n/a."""
    failures = []
    for row in collected:
        run = row.get("run") or {}
        error = str(run.get("error") or "").strip()
        status = str(run.get("status") or "")
        if status == "failed" or error:
            label = row.get("label") or "run"
            failures.append(f"{label}: {error or status}")
    if failures:
        report = dict(report)
        report["verdict"] = "fail"
        report["failures"] = failures
    return report


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
    for line in result.get("failures") or []:
        print(line, file=sys.stderr)
    for variant in result.get("variants") or []:
        note = str(variant.get("decision_note") or "").strip()
        if note:
            print(note)
    print(f"Verdict: {result.get('verdict')}")
    files = result.get("files") or {}
    for kind in ("json", "markdown", "html"):
        if files.get(kind):
            print(f"{kind}: {files[kind]}")
    sys.exit(1 if result.get("verdict") == "fail" else 0)
