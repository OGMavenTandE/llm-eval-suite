"""Isolate peak VRAM for one quantization variant.

Ollama keeps a model resident after a run. The next variant's whole-card
peak then includes that leftover. Before an Ollama variant this unloads
resident models and records the card baseline. Other local servers use
per-process memory of compute apps that were not already running.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from llm_eval.edge.hardware import probe_gpu, query_compute_apps

METHOD_CARD_AFTER_UNLOAD = "card peak after unloading other models"
METHOD_PER_PROCESS = "per-process"
METHOD_CARD = "card"


def _read_response(response) -> bytes:
    if hasattr(response, "__enter__"):
        with response as handle:
            return handle.read()
    return response.read()


def _open_bytes(url: str, payload: dict | None, opener, timeout: float) -> bytes:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    open_fn = opener or urllib.request.urlopen
    return _read_response(open_fn(request, timeout=timeout))


def unload_ollama_models(base_url: str, *, opener=None, timeout: float = 2.0) -> list[str]:
    """Ask Ollama to drop every loaded model (``keep_alive`` 0). Returns names."""
    root = base_url.rstrip("/")
    listed = json.loads(_open_bytes(root + "/api/ps", None, opener, timeout).decode("utf-8") or "{}")
    names = []
    for row in listed.get("models") or []:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name") or row.get("model") or "").strip()
        if name:
            names.append(name)
    for name in names:
        # The generate route may stream. A successful response is enough.
        _open_bytes(root + "/api/generate", {"model": name, "keep_alive": 0}, opener, timeout)
    return names


def default_process_reader() -> list[dict] | None:
    """Compute-app VRAM from nvidia-smi. None when nvidia-smi is not installed."""
    return query_compute_apps()


def _card_vram(gpu_reader) -> tuple[float | None, str]:
    try:
        row = gpu_reader() or {}
    except Exception as exc:  # noqa: BLE001 - the label records why baseline is missing
        return None, str(exc)
    value = row.get("vram_mib")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value), ""
    return None, "GPU did not report VRAM"


def prepare_variant_vram(
    provider: str,
    base_url: str,
    *,
    gpu_reader=None,
    process_reader=None,
    unloader=None,
    sleeper=None,
) -> dict:
    """Choose a VRAM method, record a baseline, and name what was unloaded.

    Ollama unloads resident models, pauses briefly so the driver can release
    them, then samples card VRAM. That card peak is the variant's peak.
    Other providers sample compute apps that appear after this call.
    """
    gpu_reader = gpu_reader or probe_gpu
    if process_reader is None:
        process_reader = default_process_reader
    if sleeper is None:
        import time

        sleeper = time.sleep
    note = ""
    unloaded: list[str] = []
    sample = "card"
    method = METHOD_CARD
    exclude: list[str] = []
    if provider == "ollama":
        unloader = unloader or unload_ollama_models
        try:
            unloaded = [str(name) for name in unloader(base_url)]
            method = METHOD_CARD_AFTER_UNLOAD
            if unloaded:
                note = "Unloaded " + ", ".join(unloaded) + "."
            else:
                note = "No other Ollama models were loaded."
        except Exception as exc:  # noqa: BLE001 - a failed unload is labeled, the variant still runs
            method = METHOD_CARD
            note = f"Could not unload Ollama models: {exc}"
    else:
        apps = None
        try:
            apps = process_reader()
        except Exception as exc:  # noqa: BLE001 - fall back to the whole card and say so
            apps = None
            note = f"Could not read per-process VRAM: {exc}"
        if isinstance(apps, list):
            method = METHOD_PER_PROCESS
            sample = "per-process"
            exclude = [str(row.get("pid")) for row in apps if row.get("pid")]
            note = note or "Peak is the VRAM of compute apps that were not running before this variant."
        elif not note:
            note = "Compute-app memory was not reported. Peak is the whole card."
    if unloaded:
        sleeper(0.5)
    baseline, baseline_reason = _card_vram(gpu_reader)
    return {
        "method": method,
        "sample": sample,
        "baseline_vram_mib": baseline,
        "baseline_reason": baseline_reason,
        "unloaded": unloaded,
        "exclude_pids": exclude,
        "note": note,
    }
