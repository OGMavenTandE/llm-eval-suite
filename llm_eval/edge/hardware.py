"""Host stamp for a run: GPU, driver, CUDA, CPU, RAM, OS, and power mode.

Readers are injectable so tests never call nvidia-smi or NVML.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from typing import Callable
from urllib.parse import urlparse

from llm_eval.edge.metrics import measured, na

Probe = Callable[[], dict | None]


def endpoint_is_local(connection: dict | None) -> bool:
    """True when GPU counters on this host describe the model server."""
    connection = connection or {}
    if connection.get("cloud"):
        return False
    kind = str(connection.get("type") or "")
    if kind in {"hf", "nanogpt"}:
        return True
    url = connection.get("base_url") or ""
    if not url and kind in {"", "ollama"}:
        return True
    host = (urlparse(str(url)).hostname or "").lower()
    return host in {"", "localhost", "127.0.0.1", "::1"}


def gpu_counter_reason(connection: dict | None, gpu_present: bool) -> str:
    if not endpoint_is_local(connection):
        return "remote server; GPU counters are not on this host"
    if not gpu_present:
        return "no NVIDIA GPU detected"
    return ""


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _cuda_from_int(value) -> str:
    """NVML packs CUDA as 12040 for 12.4."""
    try:
        number = int(value)
    except (TypeError, ValueError):
        return ""
    if number <= 0:
        return ""
    major = number // 1000
    minor = (number % 1000) // 10
    return f"{major}.{minor}"


def cpu_name() -> str:
    info = "/proc/cpuinfo"
    if os.path.isfile(info):
        try:
            with open(info, encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except OSError:
            pass
    processor = platform.processor()
    if processor:
        return processor
    return platform.machine() or ""


def ram_mib_from_bytes(total: int, avail: int) -> tuple[float, float]:
    total_mib = round(int(total) / 1048576, 1)
    used_mib = round(max(0, int(total) - int(avail)) / 1048576, 1)
    return total_mib, used_mib


def ram_total_and_used() -> tuple[float | None, float | None]:
    meminfo = "/proc/meminfo"
    if os.path.isfile(meminfo):
        total_kb = None
        avail_kb = None
        try:
            with open(meminfo, encoding="utf-8", errors="replace") as handle:
                for line in handle:
                    if line.startswith("MemTotal:"):
                        total_kb = float(line.split()[1])
                    elif line.startswith("MemAvailable:"):
                        avail_kb = float(line.split()[1])
        except OSError:
            return None, None
        if total_kb is None:
            return None, None
        total = round(total_kb / 1024, 1)
        if avail_kb is None:
            return total, None
        return total, round(max(0.0, total_kb - avail_kb) / 1024, 1)
    if os.name == "nt":
        return _windows_ram()
    return None, None


def _windows_ram() -> tuple[float | None, float | None]:
    try:
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatus()
        status.dwLength = ctypes.sizeof(MemoryStatus)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None, None
        return ram_mib_from_bytes(int(status.ullTotalPhys), int(status.ullAvailPhys))
    except Exception:
        return None, None


def os_power_mode() -> str:
    """Active Windows power scheme, when ``powercfg`` is on PATH."""
    if os.name != "nt" or not shutil.which("powercfg"):
        return ""
    try:
        out = subprocess.check_output(
            ["powercfg", "/getactivescheme"],
            text=True,
            timeout=3,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    match = re.search(r"\(([^)]+)\)", out or "")
    return match.group(1).strip() if match else ""


def cuda_from_banner(text: str) -> str:
    match = re.search(r"CUDA Version:\s*([0-9.]+)", text or "")
    return match.group(1) if match else ""


def parse_compute_apps(text: str) -> list[dict]:
    """Parse ``pid, process_name, used_gpu_memory`` rows from nvidia-smi."""
    rows = []
    for raw in (text or "").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 3:
            continue
        cleaned = parts[2].strip().strip("[]")
        if not cleaned or cleaned.upper() in {"N/A", "[N/A]", "NOT SUPPORTED"}:
            continue
        try:
            mem = float(cleaned)
        except ValueError:
            continue
        rows.append({"pid": parts[0], "name": parts[1], "vram_mib": mem})
    return rows


def parse_nvidia_smi_query(text: str) -> dict | None:
    """Parse one ``csv,noheader,nounits`` GPU row.

    Columns: name, driver_version, memory.used, power.draw, power.limit.
    """
    line = ""
    for raw in (text or "").splitlines():
        if raw.strip():
            line = raw.strip()
            break
    if not line:
        return None
    parts = [part.strip() for part in line.split(",")]
    if len(parts) < 2:
        return None

    def number(raw: str) -> float | None:
        cleaned = raw.strip().strip("[]")
        if not cleaned or cleaned.upper() in {"N/A", "[N/A]", "NOT SUPPORTED"}:
            return None
        try:
            return float(cleaned)
        except ValueError:
            return None

    name = parts[0]
    if not name or name.upper() == "N/A":
        return None
    return {
        "name": name,
        "driver": parts[1] if len(parts) > 1 else "",
        "vram_mib": number(parts[2]) if len(parts) > 2 else None,
        "watts": number(parts[3]) if len(parts) > 3 else None,
        "power_limit_w": number(parts[4]) if len(parts) > 4 else None,
        "cuda": "",
    }


def _run_smi(args: list[str], timeout: float = 3) -> str:
    if not shutil.which("nvidia-smi"):
        return ""
    try:
        return subprocess.check_output(
            ["nvidia-smi", *args],
            text=True,
            timeout=timeout,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return ""


def nvml_snapshot(nvml=None) -> dict | None:
    """One GPU sample from pynvml. Returns None when NVML is missing or fails."""
    if nvml is None:
        try:
            import pynvml as nvml  # type: ignore
        except ImportError:
            return None
    try:
        nvml.nvmlInit()
        handle = nvml.nvmlDeviceGetHandleByIndex(0)
        name = _text(nvml.nvmlDeviceGetName(handle))
        driver = _text(nvml.nvmlSystemGetDriverVersion())
        watts = None
        try:
            watts = float(nvml.nvmlDeviceGetPowerUsage(handle)) / 1000.0
        except Exception:
            watts = None
        vram = None
        try:
            mem = nvml.nvmlDeviceGetMemoryInfo(handle)
            vram = float(mem.used) / 1048576.0
        except Exception:
            vram = None
        limit = None
        try:
            limit = float(nvml.nvmlDeviceGetPowerManagementLimit(handle)) / 1000.0
        except Exception:
            limit = None
        cuda = ""
        try:
            cuda = _cuda_from_int(nvml.nvmlSystemGetCudaDriverVersion())
        except Exception:
            cuda = ""
    except Exception:
        return None
    if not name and watts is None and vram is None:
        return None
    return {
        "name": name,
        "driver": driver,
        "cuda": cuda,
        "watts": watts,
        "vram_mib": vram,
        "power_limit_w": limit,
    }


def query_compute_apps() -> list[dict] | None:
    """Compute-app rows, or None when nvidia-smi is not installed."""
    if not shutil.which("nvidia-smi"):
        return None
    text = _run_smi(
        [
            "--query-compute-apps=pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    return parse_compute_apps(text)


def smi_snapshot() -> dict | None:
    text = _run_smi(
        [
            "--query-gpu=name,driver_version,memory.used,power.draw,power.limit",
            "--format=csv,noheader,nounits",
        ]
    )
    parsed = parse_nvidia_smi_query(text)
    if parsed is None:
        return None
    banner = _run_smi([])
    parsed["cuda"] = cuda_from_banner(banner)
    return parsed


def probe_gpu(*, nvml_reader: Probe | None = None, smi_reader: Probe | None = None) -> dict | None:
    """Prefer NVML. Fall back to nvidia-smi. None means no NVIDIA GPU."""
    nvml_reader = nvml_reader or nvml_snapshot
    smi_reader = smi_reader or smi_snapshot
    try:
        nvml_row = nvml_reader()
    except Exception:
        nvml_row = None
    if nvml_row and (nvml_row.get("name") or nvml_row.get("watts") is not None or nvml_row.get("vram_mib") is not None):
        if not nvml_row.get("cuda"):
            try:
                smi_row = smi_reader()
            except Exception:
                smi_row = None
            if smi_row and smi_row.get("cuda"):
                nvml_row["cuda"] = smi_row["cuda"]
            if smi_row and nvml_row.get("power_limit_w") is None and smi_row.get("power_limit_w") is not None:
                nvml_row["power_limit_w"] = smi_row["power_limit_w"]
        return nvml_row
    try:
        return smi_reader()
    except Exception:
        return None


def _power_mode_text(gpu: dict | None, scheme: str) -> dict:
    if gpu and gpu.get("power_limit_w") is not None:
        limit = float(gpu["power_limit_w"])
        text = f"{limit:g} W limit"
        if scheme:
            text = f"{text}, {scheme}"
        return measured(text)
    if scheme:
        return measured(scheme)
    return na("power mode was not reported")


def hardware_stamp(
    connection: dict | None = None,
    *,
    probe: Probe | None = None,
    cpu: str | None = None,
    ram_total: float | None = None,
    os_name: str | None = None,
    power_scheme: str | None = None,
) -> dict:
    """Stamp this host. GPU fields are n/a, with a reason, when no NVIDIA GPU is present.

    A remote model server still gets a local CPU, RAM, and OS stamp. GPU
    counters for that run are marked remote separately by the power sampler.
    """
    gpu = probe() if probe is not None else probe_gpu()
    present = bool(gpu and (gpu.get("name") or gpu.get("driver")))
    if cpu is None:
        cpu = cpu_name()
    if ram_total is None and probe is None:
        ram_total, _used = ram_total_and_used()
    if os_name is None:
        os_name = platform.platform()
    if power_scheme is None and probe is None:
        power_scheme = os_power_mode()
    scheme = power_scheme or ""
    if present:
        gpu_cell = measured(gpu.get("name") or "NVIDIA GPU")
        driver_cell = measured(gpu.get("driver")) if gpu.get("driver") else na("driver was not reported")
        cuda_cell = measured(gpu.get("cuda")) if gpu.get("cuda") else na("CUDA version was not reported")
    else:
        reason = "no NVIDIA GPU detected"
        gpu_cell = na(reason)
        driver_cell = na(reason)
        cuda_cell = na(reason)
    return {
        "gpu": gpu_cell,
        "driver": driver_cell,
        "cuda": cuda_cell,
        "cpu": measured(cpu) if cpu else na("CPU name was not reported"),
        "ram_mib": measured(ram_total, 1) if ram_total is not None else na("RAM was not readable"),
        "os": measured(os_name) if os_name else na("OS was not reported"),
        "power_mode": _power_mode_text(gpu if present else None, scheme),
        "endpoint": "local" if endpoint_is_local(connection) else "remote",
        "gpu_present": present,
    }
