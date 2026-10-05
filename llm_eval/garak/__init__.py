"""Offline garak-shaped fixture reports for the llm-eval-suite crib.

This package does not import the NVIDIA garak package, and it does not call a
model. See docs/garak-fixture-eval.md.
"""

from llm_eval.garak.schema import SmokeReport
from llm_eval.garak.smoke import run_smoke

__all__ = ["SmokeReport", "run_smoke"]
