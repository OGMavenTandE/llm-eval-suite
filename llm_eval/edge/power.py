"""Background samples of RAM, VRAM, and GPU power.

NVML is used when pynvml imports. Otherwise the sampler shells out to
nvidia-smi. Both paths are replaceable in tests.
"""

from __future__ import annotations

import threading
import time

from llm_eval.edge.hardware import endpoint_is_local, gpu_counter_reason, probe_gpu, ram_total_and_used
from llm_eval.edge.metrics import summarize_power


class ResourceSampler:
    """Polls on a daemon thread. ``stop`` joins it and returns the raw rows."""

    def __init__(
        self,
        *,
        gpu_enabled: bool,
        gpu_reason: str,
        gpu_reader=None,
        ram_reader=None,
        interval: float = 0.5,
        clock=None,
    ):
        self.gpu_enabled = bool(gpu_enabled)
        self.gpu_reason = gpu_reason
        self.gpu_reader = gpu_reader or probe_gpu
        self.ram_reader = ram_reader or _used_ram
        self.interval = interval
        self.clock = clock or time.perf_counter
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._sample()
        self._thread = threading.Thread(target=self._loop, name="edge-sampler", daemon=True)
        self._thread.start()

    def stop(self) -> list[dict]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None
        return list(self.samples)

    def summary(self, output_tokens: int) -> dict:
        return summarize_power(
            self.samples,
            gpu_enabled=self.gpu_enabled,
            gpu_reason=self.gpu_reason,
            output_tokens=output_tokens,
        )

    def _loop(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def _sample(self) -> None:
        ram = None
        try:
            ram = self.ram_reader()
        except Exception:
            ram = None
        watts = None
        vram = None
        if self.gpu_enabled and self.gpu_reader is not None:
            try:
                row = self.gpu_reader() or {}
            except Exception:
                row = {}
            watts = row.get("watts")
            vram = row.get("vram_mib")
        with self._lock:
            self.samples.append(
                {
                    "t": float(self.clock()),
                    "ram_mib": None if ram is None else float(ram),
                    "watts": None if watts is None else float(watts),
                    "vram_mib": None if vram is None else float(vram),
                }
            )


def _used_ram() -> float | None:
    _total, used = ram_total_and_used()
    return used


def sampler_for_connection(connection: dict | None, *, probe=None, **kwargs) -> ResourceSampler:
    """GPU polling stays off for a remote server and for a host with no NVIDIA GPU."""
    gpu = probe() if probe is not None else probe_gpu()
    present = bool(gpu and (gpu.get("name") or gpu.get("vram_mib") is not None or gpu.get("watts") is not None))
    reason = gpu_counter_reason(connection, present)
    enabled = reason == "" and present

    def reader():
        if probe is not None:
            return probe()
        return probe_gpu()

    return ResourceSampler(
        gpu_enabled=enabled,
        gpu_reason=reason or ("no NVIDIA GPU detected" if not present else ""),
        gpu_reader=reader if enabled else None,
        **kwargs,
    )
