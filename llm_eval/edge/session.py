"""Collect edge metrics around one run.

The scored suite stays sequential. ``concurrency`` above 1 adds a short
parallel probe after the suite and reports that probe as the headline
latency, TTFT, and tokens/s. Power and memory cover the whole session.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

from llm_eval.edge.hardware import hardware_stamp
from llm_eval.edge.metrics import combine_summary, metric_number, sample_from_response, summarize_requests
from llm_eval.edge.power import sampler_for_connection

_STREAM_TYPES = {"OllamaModel", "OpenAIModel", "VllmModel", "LlamaCppModel"}
_PROBE_PROMPT = "Reply with the single word pong."


def edge_concurrency(expanded: dict | None, connection: dict | None) -> int:
    expanded = expanded or {}
    connection = connection or {}
    # An explicit connection value overrides the preset, so a quant compare
    # can raise concurrency without editing the preset file.
    if connection.get("edge_concurrency") not in (None, ""):
        raw = connection.get("edge_concurrency")
    else:
        raw = (expanded.get("edge") or {}).get("concurrency", 1)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return 1
    return max(1, value)


def _accepts_edge_stream(model) -> bool:
    return type(model).__name__ in _STREAM_TYPES


class EdgeSession:
    def __init__(self, connection: dict | None = None, *, concurrency: int = 1, sampler=None, hardware: dict | None = None):
        self.connection = connection or {}
        self.concurrency = max(1, int(concurrency or 1))
        self.samples: list[dict] = []
        self.load_samples: list[dict] = []
        self.load_errors: list[str] = []
        self._bucket = self.samples
        self._model = None
        self._original = None
        self.hardware = hardware if hardware is not None else hardware_stamp(self.connection)
        self.sampler = sampler if sampler is not None else sampler_for_connection(self.connection)
        self._summary: dict | None = None

    def wrap(self, model):
        """Record every ``generate`` on this model. Streaming is requested only for the HTTP clients."""
        self._model = model
        original = model.generate
        self._original = original
        session = self

        def generate(prompt, **kwargs):
            kwargs = dict(kwargs)
            if _accepts_edge_stream(model) and kwargs.get("edge_stream") is not False:
                kwargs["edge_stream"] = True
            started = time.perf_counter()
            response = original(prompt, **kwargs)
            wall_ms = (time.perf_counter() - started) * 1000
            session._bucket.append(sample_from_response(response, wall_ms))
            return response

        model.generate = generate
        return model

    def start(self) -> None:
        self.sampler.start()

    def probe(self) -> None:
        """Parallel pong calls at the configured concurrency. Skipped at 1."""
        if self.concurrency <= 1 or self._model is None:
            return
        previous = self._bucket
        self._bucket = self.load_samples

        def one():
            try:
                self._model.generate(_PROBE_PROMPT, max_tokens=8)
            except Exception as exc:  # noqa: BLE001 - recorded, the scored run already finished
                self.load_errors.append(str(exc))

        try:
            with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
                list(pool.map(lambda _index: one(), range(self.concurrency)))
        finally:
            self._bucket = previous

    def finish(self) -> dict:
        if self._summary is not None:
            return self._summary
        try:
            self.sampler.stop()
        except Exception:
            pass
        serial = summarize_requests(self.samples, concurrency=1, source="evaluation")
        if self.concurrency > 1:
            headline = summarize_requests(self.load_samples, concurrency=self.concurrency, source="load_probe")
            kept_serial = serial
        else:
            headline = serial
            kept_serial = None
        output_tokens = metric_number(headline.get("output_tokens")) or 0
        if self.concurrency > 1:
            serial_tokens = metric_number(serial.get("output_tokens")) or 0
            output_tokens = int(output_tokens) + int(serial_tokens)
        try:
            power = self.sampler.summary(int(output_tokens))
        except Exception as exc:
            power = {
                "peak_ram_mib": {"value": "n/a", "reason": str(exc)},
                "peak_vram_mib": {"value": "n/a", "reason": str(exc)},
                "power_watts_average": {"value": "n/a", "reason": str(exc)},
                "power_watts_peak": {"value": "n/a", "reason": str(exc)},
                "joules": {"value": "n/a", "reason": str(exc)},
                "joules_per_output_token": {"value": "n/a", "reason": str(exc)},
            }
        summary = combine_summary(headline, power, self.hardware, serial=kept_serial)
        if self.load_errors:
            summary["load_errors"] = list(self.load_errors)
        self._summary = summary
        return summary
