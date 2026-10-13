"""Edge performance metrics for a disconnected run.

Connectors can leave ``ttft_ms``, ``output_tokens``, and ``eval_seconds`` on
``ModelResponse.metadata``. Ollama, OpenAI, vLLM, and llama.cpp honor
``edge_stream=True``. The OpenAI-compatible servers use their own SSE reader.
"""

from llm_eval.edge.metrics import format_metric, scorecard_edge
from llm_eval.edge.session import EdgeSession, edge_concurrency

__all__ = [
    "EdgeSession",
    "edge_concurrency",
    "format_metric",
    "scorecard_edge",
]
