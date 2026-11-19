"""Gated small-model specialization.

Tools for walking one workload down the Enterprise AI Adoption Curve:
seven stages, with an evaluation gate between each pair of stages.

- Gate 1: mission metrics before seats (the gate spec)
- Gate 2: independent quality measurement (the baseline run)
- Gate 3: right-sizing log audit (``gates.audit``)
- Gate 4: workflow and agent harness (``gates.router``)
- Gate 5: acceptance against the model it would replace (``gates.acceptance``)
- Gate 6: ATO-aware continuous test and evaluation (``gates.drift``)

Every gate ends in a decision (GO, NO-GO, or INVALID) with its reasons,
written by ``gates.report``.
"""

__version__ = "0.1.0"

GO = "GO"
NO_GO = "NO-GO"
INVALID = "INVALID"
