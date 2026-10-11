"""Run one generated program in a short-lived subprocess.

Linux uses resource limits. Windows uses a job object when the platform
allows it. Both block new sockets inside the child and enforce a wall-clock
timeout from the parent. Neither opens a network namespace. On Windows that
is a process and job-object boundary, not a hypervisor.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

CHILD = Path(__file__).with_name("sandbox_child.py")


# Plus tests include a long range sum and a combinations listing. 5 seconds
# and 512 MB are not enough for those reference solutions. The ceiling is
# 20 seconds and 1 GB. cpu_s matches the wall clock so RLIMIT_CPU does not
# stop a run the parent would still allow.
DEFAULT_TIMEOUT_S = 20.0
DEFAULT_MEMORY_MB = 1024
DEFAULT_CPU_S = 20


@dataclass(frozen=True)
class RunLimits:
    timeout_s: float = DEFAULT_TIMEOUT_S
    memory_mb: int = DEFAULT_MEMORY_MB
    cpu_s: int = DEFAULT_CPU_S


@dataclass(frozen=True)
class SandboxResult:
    ok: bool
    error: str
    timed_out: bool
    isolation: str
    blocked_network: bool


def describe_isolation() -> str:
    if sys.platform == "win32":
        return (
            "Windows isolation is a job object when the process can create one "
            "(process memory, an active-process cap, kill on close), plus a "
            "wall-clock timeout and a socket monkeypatch. It is not a hypervisor "
            "and it is not a network namespace. If the job object cannot be "
            "assigned, the child falls back to the timeout and the socket block."
        )
    return (
        "Linux isolation is a subprocess with resource limits for address space, "
        "CPU time, and file size, plus a wall-clock timeout and a socket "
        "monkeypatch. It does not create a network namespace."
    )


def _child_env() -> dict[str, str]:
    kept = {}
    for key in ("PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "HOME"):
        value = os.environ.get(key)
        if value:
            kept[key] = value
    kept["PYTHONIOENCODING"] = "utf-8"
    kept["PYTHONNOUSERSITE"] = "1"
    kept["PYTHONDONTWRITEBYTECODE"] = "1"
    kept["PYTHONSAFEPATH"] = "1"
    return kept


def run_code(code: str, limits: RunLimits | None = None) -> SandboxResult:
    bounds = limits or RunLimits()
    payload = json.dumps(
        {
            "code": code,
            "memory_mb": int(bounds.memory_mb),
            "cpu_s": int(bounds.cpu_s),
        }
    ).encode("utf-8")
    timeout = max(0.2, float(bounds.timeout_s))
    try:
        # A fresh directory, not the package tree. The child cannot read or
        # write the benchmark sources by relative path.
        with tempfile.TemporaryDirectory(prefix="standard-bench-") as work:
            completed = subprocess.run(
                [sys.executable, "-I", str(CHILD)],
                input=payload,
                capture_output=True,
                timeout=timeout,
                env=_child_env(),
                cwd=work,
                check=False,
            )
    except subprocess.TimeoutExpired:
        return SandboxResult(
            ok=False,
            error="timed out",
            timed_out=True,
            isolation=describe_isolation(),
            blocked_network=False,
        )
    stdout = completed.stdout.decode("utf-8", errors="replace").strip()
    line = stdout.splitlines()[-1] if stdout else ""
    try:
        report = json.loads(line) if line else {}
    except json.JSONDecodeError:
        report = {}
    if not isinstance(report, dict):
        report = {}
    error = str(report.get("error") or "")
    if not error and completed.returncode != 0:
        err = completed.stderr.decode("utf-8", errors="replace").strip()
        error = err[-500:] or f"exit {completed.returncode}"
    return SandboxResult(
        ok=bool(report.get("ok")),
        error=error,
        timed_out=False,
        isolation=str(report.get("isolation") or describe_isolation()),
        blocked_network=bool(report.get("blocked_network")),
    )
