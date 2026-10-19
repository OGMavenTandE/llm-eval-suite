"""Decision-to-evidence map, phases 1 and 2.

The tool reads a YAML map and click-through ``run.json`` files, then computes
an evidence status for each row: met, partial, gap, or not assessed. It does
not grant authority. The only authority text on the page is the optional
human field ``recommended_authority``, shown verbatim.

Trigger rows are displayed and not evaluated. The click-through app is not
wired to this package.
"""

from llm_eval.decision_map.providers import (
    ExternalMetricProvider,
    MetricReading,
    MetricRegistry,
    register_metric,
)
from llm_eval.decision_map.render import render_html
from llm_eval.decision_map.resolve import resolve_map
from llm_eval.decision_map.schema import DecisionMap

__all__ = [
    "DecisionMap",
    "ExternalMetricProvider",
    "MetricReading",
    "MetricRegistry",
    "register_metric",
    "render_html",
    "resolve_map",
]
