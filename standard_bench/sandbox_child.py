"""Child process for one code-eval program.

Invoked by path, not as an import. Applies resource limits, blocks sockets,
then execs the program from stdin JSON. Prints one JSON object on stdout.
"""

from __future__ import annotations

import json
import sys


def _block_network() -> None:
    import socket

    def _blocked(*_args, **_kwargs):
        raise OSError("network disabled in the eval sandbox")

    class _BlockedSocket(socket.socket):
        def connect(self, _address):  # noqa: ANN001
            raise OSError("network disabled in the eval sandbox")

        def connect_ex(self, _address):  # noqa: ANN001
            raise OSError("network disabled in the eval sandbox")

    socket.socket = _BlockedSocket
    socket.create_connection = _blocked
    if hasattr(socket, "create_server"):
        socket.create_server = _blocked
    socket.getaddrinfo = _blocked
    # `import socket` is the Python wrapper. `import _socket` is the C module
    # underneath it, and a program can call that constructor directly.
    c_socket = sys.modules.get("_socket")
    if c_socket is None:
        try:
            import _socket as c_socket
        except ImportError:
            c_socket = None
    if c_socket is not None:
        for name in ("socket", "create_connection", "getaddrinfo"):
            if hasattr(c_socket, name):
                try:
                    setattr(c_socket, name, _blocked)
                except (AttributeError, TypeError):
                    pass


def _linux_limits(memory_mb: int, cpu_s: int) -> str:
    import resource

    applied = []
    memory_bytes = max(64, int(memory_mb)) * 1024 * 1024
    try:
        resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
        applied.append("address-space")
    except (ValueError, OSError):
        pass
    cpu = max(1, int(cpu_s))
    try:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
        applied.append("cpu")
    except (ValueError, OSError):
        pass
    try:
        resource.setrlimit(resource.RLIMIT_FSIZE, (8 * 1024 * 1024, 8 * 1024 * 1024))
        applied.append("file-size")
    except (ValueError, OSError):
        pass
    try:
        # 32 is below what CPython needs to start the child. 128 covers
        # the interpreter and still caps a runaway open loop.
        resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
        applied.append("files")
    except (ValueError, OSError):
        pass
    if not applied:
        return "socket-only"
    return "rlimit+" + "+".join(applied)


def _windows_job(memory_mb: int) -> str:
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    process_memory = 0x00000100
    job_memory = 0x00000200
    active_process = 0x00000008
    kill_on_close = 0x00002000
    job_extended = 9

    class IoCounters(ctypes.Structure):
        _fields_ = [
            ("ReadOperationCount", ctypes.c_uint64),
            ("WriteOperationCount", ctypes.c_uint64),
            ("OtherOperationCount", ctypes.c_uint64),
            ("ReadTransferCount", ctypes.c_uint64),
            ("WriteTransferCount", ctypes.c_uint64),
            ("OtherTransferCount", ctypes.c_uint64),
        ]

    class BasicLimit(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", ctypes.c_int64),
            ("PerJobUserTimeLimit", ctypes.c_int64),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class ExtendedLimit(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", BasicLimit),
            ("IoInfo", IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        return "timeout-and-socket"
    memory_bytes = max(64, int(memory_mb)) * 1024 * 1024
    info = ExtendedLimit()
    flags = process_memory | job_memory | kill_on_close | active_process
    info.BasicLimitInformation.LimitFlags = flags
    info.ProcessMemoryLimit = memory_bytes
    info.JobMemoryLimit = memory_bytes
    info.BasicLimitInformation.ActiveProcessLimit = 8
    ok = kernel32.SetInformationJobObject(job, job_extended, ctypes.byref(info), ctypes.sizeof(info))
    if not ok:
        info.BasicLimitInformation.LimitFlags = process_memory | job_memory | kill_on_close
        info.BasicLimitInformation.ActiveProcessLimit = 0
        ok = kernel32.SetInformationJobObject(job, job_extended, ctypes.byref(info), ctypes.sizeof(info))
    if not ok:
        return "timeout-and-socket"
    assigned = kernel32.AssignProcessToJobObject(job, kernel32.GetCurrentProcess())
    if not assigned:
        return "timeout-and-socket"
    return "windows-job-object"


def main() -> int:
    raw = sys.stdin.buffer.read()
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        sys.stdout.write(json.dumps({"ok": False, "error": f"bad payload: {exc}", "isolation": ""}))
        return 1
    code = str(payload.get("code") or "")
    memory_mb = int(payload.get("memory_mb") or 512)
    cpu_s = int(payload.get("cpu_s") or 2)
    _block_network()
    if sys.platform == "win32":
        isolation = _windows_job(memory_mb)
    else:
        isolation = _linux_limits(memory_mb, cpu_s)
    namespace: dict = {"__name__": "__main__"}
    try:
        exec(compile(code, "<eval>", "exec"), namespace, namespace)  # noqa: S102 - the sandbox is the boundary
    except BaseException as exc:  # noqa: BLE001 - the parent needs the error class
        message = f"{type(exc).__name__}: {exc}"
        blocked = "network disabled" in message
        sys.stdout.write(
            json.dumps(
                {
                    "ok": False,
                    "error": message[:500],
                    "isolation": isolation,
                    "blocked_network": blocked,
                }
            )
        )
        return 1
    sys.stdout.write(json.dumps({"ok": True, "error": "", "isolation": isolation, "blocked_network": False}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
