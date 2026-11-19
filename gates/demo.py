"""A simulated end-to-end packet for the message-triage reference case.

The predictions here are SIMULATED: they are the true labels with seeded,
label-dependent errors, not output from any model. They exist so the gate
tooling, tests and report layout can be checked without a GPU. Every page
written by the demo carries a banner saying so. Do not quote its numbers.
For real results, run ``gates predict`` with real models (see docs/gates.md).
"""

from __future__ import annotations

import hashlib
import random
from pathlib import Path

from gates.acceptance import gate2_baseline, gate5_acceptance
from gates.audit import load_audit_config, run_audit
from gates.data import load_items, read_jsonl
from gates.drift import run_drift
from gates.report import file_digest, write_gate, write_packet_index
from gates.router import run_router_harness
from gates.spec import load_spec

SIMULATED = ("SIMULATED DEMO: the inputs on this page are synthetic, and any predictions come from gates.demo, "
             "not from a model. The checks are real; the numbers are not results.")
EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "message_triage"


def _difficulty(item_id: str) -> float:
    """Same value for every simulated model, so hard items are hard for all of them."""
    digest = hashlib.sha256(item_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def simulate(items: list[dict], labels: list[str], error_by_label: dict[str, float], seed: int) -> dict[str, dict]:
    """Most errors land on shared hard items, as real model errors tend to; a fifth are independent."""
    rng = random.Random(seed)
    preds = {}
    for item in items:
        pred = item["label"]
        rate = error_by_label.get(item["label"], 0.0)
        if _difficulty(item["id"]) < rate * 0.8 or rng.random() < rate * 0.2:
            pred = rng.choice([label for label in labels if label != item["label"]])
        preds[item["id"]] = {"id": item["id"], "prediction": pred, "simulated": True}
    return preds


def run_demo(out_dir: str | Path, example: Path = EXAMPLE, *, generated_at: str | None = None) -> Path:
    out = Path(out_dir)
    spec_path, audit_path = example / "spec.yaml", example / "audit.yaml"
    items_path, shift_path = example / "items.jsonl", example / "shift_items.jsonl"
    spec = load_spec(spec_path)
    audit = load_audit_config(audit_path)
    test = load_items(items_path, split="test")
    shift = load_items(shift_path)

    baseline = simulate(test, spec.labels, {"FLASH": 0.0, "IMMEDIATE": 0.02, "PRIORITY": 0.05, "ROUTINE": 0.04}, 1)
    candidate = simulate(test, spec.labels, {"FLASH": 0.0, "IMMEDIATE": 0.03, "PRIORITY": 0.05, "ROUTINE": 0.03}, 2)
    current = simulate(shift, spec.labels, {"FLASH": 0.0, "IMMEDIATE": 0.12, "PRIORITY": 0.06, "ROUTINE": 0.03}, 3)
    reference = simulate(shift, spec.labels, {"FLASH": 0.0, "IMMEDIATE": 0.03, "PRIORITY": 0.05, "ROUTINE": 0.03}, 2)

    decisions = [
        gate2_baseline(spec, test, baseline),
        run_audit(read_jsonl(example / "traffic_log.jsonl"), audit),
        run_router_harness(read_jsonl(example / "router_cases.jsonl"), audit),
        gate5_acceptance(spec, test, candidate, baseline, candidate_name="specialized small model"),
        run_drift(spec, reference, current, items=shift, reference_name="accepted model",
                  current_name="after base-model update"),
    ]
    shared_inputs = {"spec": file_digest(spec_path), "items": file_digest(items_path)}
    per_gate = {
        2: shared_inputs,
        3: {"audit config": file_digest(audit_path), "traffic log": file_digest(example / "traffic_log.jsonl")},
        4: {"audit config": file_digest(audit_path), "router cases": file_digest(example / "router_cases.jsonl")},
        5: shared_inputs,
        6: {"spec": file_digest(spec_path), "monitoring items": file_digest(shift_path)},
    }
    for d in decisions:
        d.notice = SIMULATED
        d.inputs = per_gate[d.gate]
        write_gate(d, out, generated_at=generated_at)
    return write_packet_index(out, spec.workload + " (simulated demo)")
