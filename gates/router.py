"""Gate 4: workflow and agent harness.

Tests the routing rules before any model is swapped in. Each test case names
the route it should take. Stress cases are the ones that matter most: a hard
task that must not land on the small model, and an easy one that should not
stay on the large one.
"""

from __future__ import annotations

from gates.acceptance import Check, GateDecision
from gates.audit import AuditConfig, sort_row


def run_router_harness(cases: list[dict], config: AuditConfig, *, min_route_accuracy: float = 0.95) -> GateDecision:
    misroutes = []
    stress_total = stress_right = right = 0
    unsafe = 0  # high-reasoning or high-cost work sent to the small model
    for case in cases:
        if "expected_route" not in case:
            raise ValueError(f"router case is missing 'expected_route': {case.get('id')}")
        got = sort_row(case, config)
        ok = got.route == case["expected_route"]
        right += ok
        if case.get("stress"):
            stress_total += 1
            stress_right += ok
        if not ok:
            if got.route == "small" and case["expected_route"] == "large":
                unsafe += 1
            misroutes.append(
                {
                    "id": case.get("id"),
                    "expected": case["expected_route"],
                    "got": got.route,
                    "task_type": got.task_type,
                    "reasoning": got.reasoning,
                    "stress": bool(case.get("stress")),
                    "note": case.get("note", ""),
                }
            )
    n = len(cases)
    accuracy = right / n if n else 0.0
    decision = GateDecision(4, "Workflow and agent harness", workload="routing rules")
    decision.checks = [
        Check("Router test cases present", n > 0 and stress_total > 0,
              f"{n} cases, {stress_total} of them stress cases. At least one stress case is required.",
              blocks_validity=True),
        Check("No hard or high-cost task routed to the small model", unsafe == 0,
              f"{unsafe} cases that belong on the large model were routed to the small one."),
        Check("Every stress case routed as expected", stress_right == stress_total,
              f"{stress_right} of {stress_total} stress cases routed as expected."),
        Check("Overall route accuracy at or above the bar", accuracy >= min_route_accuracy,
              f"{right} of {n} cases routed as expected ({accuracy * 100:.1f}%); bar {min_route_accuracy * 100:.1f}%."),
    ]
    decision.decide()
    decision.metrics = {"cases": n, "right": right, "stress": stress_total, "stress_right": stress_right,
                        "unsafe": unsafe, "misroutes": misroutes}
    if misroutes:
        decision.findings.append(
            "Each misroute is fixed one of three ways: once, as a saved test case, or by changing the routing rules."
        )
    return decision
