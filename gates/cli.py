"""Command line for the evaluation gates.

    python -m gates baseline --spec S --items I --predictions P --out DIR       # Gate 2
    python -m gates audit    --config A --log L --out DIR                        # Gate 3
    python -m gates router   --config A --cases C --out DIR                      # Gate 4
    python -m gates predict  --spec S --items I --provider ollama --model M --out P
    python -m gates accept   --spec S --items I --candidate P1 --baseline P2 --out DIR   # Gate 5
    python -m gates drift    --spec S --reference P1 --current P2 [--items I] --out DIR  # Gate 6
    python -m gates packet   --out DIR --workload NAME
    python -m gates demo     --out DIR    # simulated, for layout and tests only

Exit code: 0 for GO, 1 for NO-GO, 2 for INVALID, so a pipeline can stop on a gate.
"""

from __future__ import annotations

import argparse
import sys

from gates import GO, NO_GO
from gates.acceptance import gate2_baseline, gate5_acceptance
from gates.audit import load_audit_config, run_audit
from gates.data import load_items, load_predictions, read_jsonl
from gates.drift import run_drift
from gates.report import file_digest, write_gate, write_packet_index
from gates.router import run_router_harness
from gates.spec import load_spec

_EXIT = {GO: 0, NO_GO: 1}


def _finish(decision, out: str, inputs: dict) -> int:
    decision.inputs = {role: file_digest(path) for role, path in inputs.items() if path}
    paths = write_gate(decision, out)
    print(f"Gate {decision.gate} {decision.title}: {decision.verdict}")
    for check in decision.checks:
        print(f"  [{'PASS' if check.passed else 'FAIL'}] {check.name}: {check.detail}")
    for finding in decision.findings:
        print(f"  - {finding}")
    print(f"  report: {paths['html']}")
    return _EXIT.get(decision.verdict, 2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gates", description="Evaluation gates for small-model specialization")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("baseline", help="Gate 2: measure the model in use today")
    p.add_argument("--spec", required=True)
    p.add_argument("--items", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--predictions", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("audit", help="Gate 3: right-sizing log audit")
    p.add_argument("--config", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("router", help="Gate 4: workflow and routing harness")
    p.add_argument("--config", required=True)
    p.add_argument("--cases", required=True)
    p.add_argument("--min-accuracy", type=float, default=0.95)
    p.add_argument("--out", required=True)

    p = sub.add_parser("predict", help="Run a model over items and save predictions")
    p.add_argument("--spec", required=True)
    p.add_argument("--items", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--provider", required=True, help="ollama, openai, vllm, llama_cpp, or hf")
    p.add_argument("--model", required=True)
    p.add_argument("--base-url")
    p.add_argument("--folder", help="Local model folder for --provider hf")
    p.add_argument("--max-new-tokens", type=int, default=16)
    p.add_argument("--out", required=True)

    p = sub.add_parser("accept", help="Gate 5: candidate against the model it would replace")
    p.add_argument("--spec", required=True)
    p.add_argument("--items", required=True)
    p.add_argument("--split", default="test")
    p.add_argument("--candidate", required=True)
    p.add_argument("--candidate-name", default="candidate")
    p.add_argument("--baseline", required=True)
    p.add_argument("--out", required=True)

    p = sub.add_parser("drift", help="Gate 6: continuous T&E after cutover")
    p.add_argument("--spec", required=True)
    p.add_argument("--reference", required=True)
    p.add_argument("--current", required=True)
    p.add_argument("--items", help="Labeled monitoring items, for the accuracy check")
    p.add_argument("--split", default=None)
    p.add_argument("--reference-name", default="reference")
    p.add_argument("--current-name", default="current")
    p.add_argument("--out", required=True)

    p = sub.add_parser("packet", help="Write PACKET.md listing every gate report in a folder")
    p.add_argument("--out", required=True)
    p.add_argument("--workload", required=True)

    p = sub.add_parser("demo", help="Simulated packet for the triage reference case (not real results)")
    p.add_argument("--out", required=True)

    args = parser.parse_args(argv)

    if args.cmd == "baseline":
        spec = load_spec(args.spec)
        items = load_items(args.items, split=args.split)
        preds = load_predictions(args.predictions, spec.labels)
        return _finish(gate2_baseline(spec, items, preds), args.out,
                       {"spec": args.spec, "items": args.items, "baseline predictions": args.predictions})
    if args.cmd == "audit":
        return _finish(run_audit(read_jsonl(args.log), load_audit_config(args.config)), args.out,
                       {"audit config": args.config, "traffic log": args.log})
    if args.cmd == "router":
        decision = run_router_harness(read_jsonl(args.cases), load_audit_config(args.config),
                                      min_route_accuracy=args.min_accuracy)
        return _finish(decision, args.out, {"audit config": args.config, "router cases": args.cases})
    if args.cmd == "predict":
        from gates.predict import build_model, predict

        spec = load_spec(args.spec)
        params = {"max_new_tokens": args.max_new_tokens}
        if args.base_url:
            params["base_url"] = args.base_url
        if args.folder:
            params["folder"] = args.folder
        model = build_model(args.provider, args.model, params)
        predict(spec, load_items(args.items, split=args.split), model, out_path=args.out)
        print(f"wrote {args.out}")
        return 0
    if args.cmd == "accept":
        spec = load_spec(args.spec)
        items = load_items(args.items, split=args.split)
        decision = gate5_acceptance(spec, items, load_predictions(args.candidate, spec.labels),
                                    load_predictions(args.baseline, spec.labels), candidate_name=args.candidate_name)
        return _finish(decision, args.out, {"spec": args.spec, "items": args.items,
                                            "candidate predictions": args.candidate,
                                            "baseline predictions": args.baseline})
    if args.cmd == "drift":
        spec = load_spec(args.spec)
        items = load_items(args.items, split=args.split) if args.items else None
        decision = run_drift(spec, load_predictions(args.reference, spec.labels),
                             load_predictions(args.current, spec.labels), items=items,
                             reference_name=args.reference_name, current_name=args.current_name)
        return _finish(decision, args.out, {"spec": args.spec, "monitoring items": args.items,
                                            "reference predictions": args.reference,
                                            "current predictions": args.current})
    if args.cmd == "packet":
        print(write_packet_index(args.out, args.workload))
        return 0
    if args.cmd == "demo":
        from gates.demo import run_demo

        print(run_demo(args.out))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
