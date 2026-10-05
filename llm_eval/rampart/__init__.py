"""Offline RAMPART-shaped smoke records for the llm-eval-suite crib.

This package does not import the microsoft/RAMPART package, and it does not
call an LLM. See docs/rampart-pytest-smoke.md.
"""

from llm_eval.rampart.schema import SmokeReport
from llm_eval.rampart.smoke import run_smoke

__all__ = ["SmokeReport", "run_smoke"]
