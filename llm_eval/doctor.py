"""Air-gap self-check.

``python -m llm_eval doctor`` prints one PASS or FAIL line per check and an
OVERALL line. The process exits 0 only when every check passes.

Egress is a real socket connect to a public address. A timeout or a refused
route is a pass for a closed network. Proxy variables and a package-index
URL fail the env check because either one would send traffic off the machine.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from llm_eval.bundle_offline import BUNDLE_DIRNAME, verify_sha256sums
from llm_eval.offline import DETECTOR_PINS

EGRESS_HOST = "1.1.1.1"
EGRESS_PORT = 443
EGRESS_TIMEOUT = 2.0
PROXY_ENV = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "SOCKS_PROXY",
    "socks_proxy",
    "PIP_INDEX_URL",
    "PIP_EXTRA_INDEX_URL",
)


class DoctorError(RuntimeError):
    """The doctor could not start the check. This is not a failed check line."""


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str


def probe_egress(host: str = EGRESS_HOST, port: int = EGRESS_PORT, timeout: float = EGRESS_TIMEOUT) -> tuple[bool, str]:
    """Return ``(blocked, detail)``. Blocked means no egress."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect((host, port))
    except OSError as exc:
        return True, f"no route to {host}:{port} ({exc.__class__.__name__})"
    else:
        return False, f"connected to {host}:{port}"
    finally:
        sock.close()


def proxy_env_problems(environ: dict | None = None) -> list[str]:
    source = os.environ if environ is None else environ
    found = []
    for name in PROXY_ENV:
        if (source.get(name) or "").strip():
            found.append(name)
    return found


def default_gpu_name() -> str | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    lines = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    return lines[0] if lines else None


def format_report(checks: list[Check]) -> str:
    lines = []
    for check in checks:
        status = "PASS" if check.ok else "FAIL"
        detail = " ".join(check.detail.split())
        lines.append(f"{status}  {check.name:<12} {detail}")
    overall = "PASS" if checks and all(check.ok for check in checks) else "FAIL"
    lines.append(f"OVERALL {overall}")
    return "\n".join(lines) + "\n"


def _safe_extract(archive: tarfile.TarFile, dest: Path) -> None:
    root = dest.resolve()
    for member in archive.getmembers():
        target = (root / member.name).resolve()
        if target != root and root not in target.parents:
            raise DoctorError(f"Refusing to extract {member.name}.")
    kwargs = {"filter": "data"} if sys.version_info >= (3, 12) else {}
    archive.extractall(root, **kwargs)


def open_bundle(path: Path) -> tuple[Path, TemporaryDirectory | None]:
    if path.is_dir():
        return path, None
    if not path.is_file():
        raise DoctorError(f"No bundle at {path}.")
    if not (path.name.endswith(".tar.gz") or path.name.endswith(".tgz")):
        raise DoctorError(f"Bundle must be a directory or a .tar.gz file, not {path.name}.")
    temp = TemporaryDirectory(prefix="llm-eval-doctor-")
    try:
        with tarfile.open(path, "r:gz") as archive:
            _safe_extract(archive, Path(temp.name))
    except (tarfile.TarError, OSError) as exc:
        temp.cleanup()
        raise DoctorError(f"Could not read {path}: {exc}") from exc
    unpacked = Path(temp.name) / BUNDLE_DIRNAME
    if not unpacked.is_dir():
        children = [child for child in Path(temp.name).iterdir() if child.is_dir()]
        unpacked = children[0] if len(children) == 1 else Path(temp.name)
    return unpacked, temp


def _detector_root(bundle: Path | None, explicit: Path | None) -> Path | None:
    if explicit is not None:
        return explicit
    if bundle is not None and (bundle / "garak_detectors").is_dir():
        return bundle / "garak_detectors"
    return None


def _check_detectors(folder: Path | None) -> Check:
    missing = []
    for pin in DETECTOR_PINS:
        place = (folder / pin.key) if folder is not None else pin.directory()
        absent = [name for name in pin.files if not (place / name).is_file()]
        if absent:
            missing.append(f"{pin.key} missing {', '.join(absent)}")
    if missing:
        return Check("detectors", False, "; ".join(missing))
    return Check("detectors", True, f"{len(DETECTOR_PINS)} detector folders")


def judge_present(root: Path, name: str) -> bool:
    if not root.is_dir():
        return False
    safe = name.replace(":", "-").replace("/", "-")
    for candidate in (root / safe, root / "models" / safe):
        if candidate.is_file():
            return True
        if candidate.is_dir() and any(path.is_file() for path in candidate.rglob("*")):
            return True
    library, _, tag = name.partition(":")
    tag = tag or "latest"
    manifest = root / "manifests" / "registry.ollama.ai" / "library" / library / tag
    return manifest.is_file()


def _configured_judges(bundle: Path | None) -> list[str]:
    if bundle is not None:
        payload_path = bundle / "judges" / "JUDGES.json"
        if payload_path.is_file():
            try:
                payload = json.loads(payload_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {}
            names = []
            for row in payload.get("models") or []:
                if isinstance(row, dict) and row.get("name"):
                    names.append(str(row["name"]))
                elif isinstance(row, str) and row:
                    names.append(row)
            if names:
                return names
    from llm_eval_suite.connections import DEFAULT_JUDGES

    return [str(row["model"]) for row in DEFAULT_JUDGES]


def _judge_search_roots(bundle: Path | None, judges_dir: Path | None) -> list[Path]:
    roots: list[Path] = []
    if judges_dir is not None:
        roots.append(judges_dir)
    env_dir = (os.environ.get("LLM_EVAL_JUDGES_DIR") or "").strip()
    if env_dir:
        roots.append(Path(env_dir))
    if bundle is not None:
        roots.append(bundle / "judges")
    ollama = (os.environ.get("OLLAMA_MODELS") or "").strip()
    if ollama:
        roots.append(Path(ollama))
    else:
        roots.append(Path.home() / ".ollama" / "models")
    return roots


def _check_judges(bundle: Path | None, judges_dir: Path | None) -> Check:
    names = _configured_judges(bundle)
    roots = _judge_search_roots(bundle, judges_dir)
    missing = [name for name in names if not any(judge_present(root, name) for root in roots)]
    if missing:
        return Check("judges", False, "missing " + ", ".join(missing))
    return Check("judges", True, f"{len(names)} judges")


def _check_kit(bundle: Path | None) -> Check:
    if bundle is None:
        return Check("kit", False, "no bundle to verify")
    kit_dir = bundle / "kit"
    zips = sorted(kit_dir.glob("*.zip")) if kit_dir.is_dir() else []
    if not zips:
        return Check("kit", False, "capability kit zip is missing")
    from standard_bench.bundle import BundleError, verify_kit
    from standard_bench.manifest import HashMismatch, ManifestError

    try:
        verify_kit(zips[0])
    except (BundleError, ManifestError, HashMismatch) as exc:
        return Check("kit", False, str(exc))
    return Check("kit", True, zips[0].name)


def _check_python() -> Check:
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 10):
        return Check("python", True, version)
    return Check("python", False, f"{version} is older than 3.10")


def _check_gpu(require_gpu: bool, lookup) -> Check:
    name = lookup()
    if name:
        return Check("gpu", True, name)
    if require_gpu:
        return Check("gpu", False, "nvidia-smi did not report a GPU")
    return Check("gpu", True, "no NVIDIA GPU detected; this check does not require one")


def run_doctor(
    bundle: str | Path | None = None,
    *,
    judges_dir: str | Path | None = None,
    detectors_dir: str | Path | None = None,
    require_gpu: bool = False,
    egress_probe=None,
    gpu_lookup=None,
) -> list[Check]:
    """Run every check. The caller prints the report and chooses the exit code."""
    probe = egress_probe or probe_egress
    lookup = gpu_lookup or default_gpu_name
    holder: TemporaryDirectory | None = None
    root: Path | None = None
    if bundle is not None:
        root, holder = open_bundle(Path(bundle))
    elif (Path.cwd() / "SHA256SUMS").is_file():
        root = Path.cwd()
    try:
        blocked, egress_detail = probe()
        proxies = proxy_env_problems()
        if root is None:
            hashes = (False, "SHA256SUMS is missing")
        else:
            hashes = verify_sha256sums(root)
        checks = [
            Check("egress", blocked, egress_detail if blocked else egress_detail + "; this machine is not air-gapped"),
            Check("proxy", not proxies, "no proxy or package index in the environment" if not proxies else "set " + ", ".join(proxies)),
            Check("hashes", hashes[0], hashes[1]),
            _check_kit(root),
            _check_detectors(_detector_root(root, Path(detectors_dir) if detectors_dir else None)),
            _check_judges(root, Path(judges_dir) if judges_dir else None),
            _check_python(),
            _check_gpu(require_gpu, lookup),
        ]
        return checks
    finally:
        if holder is not None:
            holder.cleanup()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm_eval doctor",
        description="Check egress, hashes, detectors, judges, Python, and the GPU. Prints PASS or FAIL.",
    )
    parser.add_argument("--bundle", default="", help="Unpacked bundle directory or the .tar.gz from bundle-offline.")
    parser.add_argument("--judges-dir", default="", help="Directory that holds the local judge models.")
    parser.add_argument("--detectors-dir", default="", help="Parent of the garak detector folders.")
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="Fail when nvidia-smi does not report a GPU. The default records a CPU-only machine as a pass.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        checks = run_doctor(
            args.bundle or None,
            judges_dir=args.judges_dir or None,
            detectors_dir=args.detectors_dir or None,
            require_gpu=bool(args.require_gpu),
        )
    except DoctorError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    sys.stdout.write(format_report(checks))
    raise SystemExit(0 if all(check.ok for check in checks) else 1)
