"""Offline Dioptra-shaped records for the llm-eval-suite light pilot.

This package does not import or call the NIST Dioptra client, and it does not
open a network connection. See docs/dioptra-light-pilot.md.
"""

from llm_eval.dioptra.client import OfflineDioptraClient
from llm_eval.dioptra.schema import PilotBundle

__all__ = ["OfflineDioptraClient", "PilotBundle"]
