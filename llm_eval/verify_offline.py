"""Build the offline bundle and prove a clean install never touches the network.

``python -m llm_eval verify-offline`` packs wheels from the local environment,
builds the bundle with no package index and no dataset fetch, installs that
bundle into a clean virtualenv with ``pip --no-index``, and runs doctor plus
a stub quant-compare on the standard fixtures. A socket connect is recorded
and fails the run.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from llm_eval.doctor import PROXY_ENV

NETBLOCK = """\
import os
import socket

_LOG = os.environ.get("LLM_EVAL_NETBLOCK_LOG", "")


class OfflineNetworkBlocked(OSError):
    pass


def _note(address):
    if not _LOG:
        return
    try:
        with open(_LOG, "a", encoding="utf-8") as handle:
            handle.write(repr(address) + "\\n")
    except OSError:
        pass


def _refuse(address):
    _note(address)
    raise OfflineNetworkBlocked(
        "Offline verification blocked a network attempt to %s. Nothing was sent." % (address,)
    )


def _connect(self, address):
    _refuse(address)


def _connect_ex(self, address):
    _refuse(address)


def _create_connection(address, *args, **kwargs):
    _refuse(address)


socket.socket.connect = _connect
socket.socket.connect_ex = _connect_ex
socket.create_connection = _create_connection
"""

NETBLOCK_PTH = "import llm_eval_netblock\n"

EGRESS_ATTEMPT = "('1.1.1.1', 443)"
OFFLINE_ENV = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "HF_DATASETS_OFFLINE": "1",
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "LLM_EVAL_OFFLINE": "1",
    "PIP_NO_INDEX": "1",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "PYTHONNOUSERSITE": "1",
}


class VerifyOfflineError(RuntimeError):
    """The offline install check failed. The message is the reason."""


def network_violations(text: str, *, allow_egress: bool) -> list[str]:
    """Return recorded socket attempts that the check must not allow."""
    violations = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if allow_egress and line == EGRESS_ATTEMPT:
            continue
        violations.append(line)
    return violations


def _canonicalize(name: str) -> str:
    return name.lower().replace("_", "-")


def _requirement_selected(raw: str, extras: set[str]) -> bool:
    from packaging.requirements import Requirement

    req = Requirement(raw)
    if req.marker is None:
        return True
    candidates = extras or {""}
    return any(req.marker.evaluate({"extra": extra}) for extra in candidates)


def closure(roots: list[str]) -> list[str]:
    """Installed distribution names needed by ``roots``, including extras."""
    from importlib.metadata import distribution
    from packaging.requirements import Requirement

    pending = []
    for raw in roots:
        req = Requirement(raw)
        pending.append((req.name, set(req.extras)))
    seen: set[str] = set()
    ordered: list[str] = []
    while pending:
        name, extras = pending.pop()
        key = _canonicalize(name)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(name)
        dist = distribution(name)
        for raw in dist.requires or []:
            if not _requirement_selected(raw, extras):
                continue
            child = Requirement(raw)
            pending.append((child.name, set(child.extras)))
    return ordered


def project_roots() -> list[str]:
    """Runtime requirements plus the api extra. Dev and hf extras stay out."""
    from importlib.metadata import distribution
    from packaging.requirements import Requirement

    roots = []
    for raw in distribution("llm-eval-suite").requires or []:
        req = Requirement(raw)
        if req.marker is None or req.marker.evaluate({"extra": "api"}):
            roots.append(raw)
    if not roots:
        raise VerifyOfflineError("llm-eval-suite did not declare install requirements.")
    return roots


def _record_hash(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    token = base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")
    return f"sha256={token}"


def _wheel_filename(name: str, version: str) -> str:
    from packaging.utils import canonicalize_version
    from packaging.version import Version

    normalized = _canonicalize(name).replace("-", "_")
    try:
        version_text = canonicalize_version(Version(version))
    except Exception:
        version_text = version
    version_text = version_text.replace("-", "_")
    return f"{normalized}-{version_text}-py3-none-any.whl"


def _read_under(root: Path, path: Path) -> tuple[str, bytes] | None:
    if not path.is_file():
        return None
    if "__pycache__" in path.parts or path.suffix == ".pyc":
        return None
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return None
    return relative.as_posix(), path.read_bytes()


def _import_names(dist) -> list[str]:
    names: list[str] = []
    top = dist.read_text("top_level.txt") or ""
    for line in top.splitlines():
        text = line.strip()
        if text and text not in names:
            names.append(text)
    meta = str(dist.metadata["Name"] or "")
    for text in (meta, meta.lower(), meta.replace("-", "_"), meta.lower().replace("-", "_")):
        if text and text not in names:
            names.append(text)
    return names


def _fallback_payloads(dist) -> list[tuple[str, bytes]]:
    info = Path(dist._path)
    root = info.parent
    payloads: list[tuple[str, bytes]] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        item = _read_under(root, path)
        if item is None or item[0] in seen or item[0].endswith("/RECORD"):
            return
        if item[0].endswith(".dist-info/RECORD"):
            return
        seen.add(item[0])
        payloads.append(item)

    if info.is_dir():
        for path in info.rglob("*"):
            add(path)
    for name in _import_names(dist):
        package = root / name
        if package.is_dir():
            for path in package.rglob("*"):
                add(path)
        if not root.is_dir():
            continue
        for path in root.iterdir():
            if path.name == name or path.name.startswith(name + "."):
                add(path)
    return payloads


def _has_importable(payloads: list[tuple[str, bytes]]) -> bool:
    for arc, _data in payloads:
        if ".dist-info/" in arc or arc.endswith(".dist-info"):
            continue
        return True
    return False


def distribution_payloads(name: str) -> tuple[object, list[tuple[str, bytes]]]:
    from importlib.metadata import distribution

    dist = distribution(name)
    direct = dist.read_text("direct_url.json") or ""
    if '"editable"' in direct or "editable" in direct and "direct_url" in direct:
        raise VerifyOfflineError(f"{name} is an editable install. Pack the source tree instead.")
    root = Path(str(dist.locate_file(""))).resolve()
    payloads: list[tuple[str, bytes]] = []
    seen: set[str] = set()
    files = dist.files
    if files:
        for rel in files:
            text = Path(str(rel)).as_posix()
            if text.endswith("/RECORD") or text.endswith(".dist-info/RECORD"):
                continue
            if ".." in text.split("/"):
                continue
            item = _read_under(root, Path(str(dist.locate_file(rel))))
            if item is None or item[0] in seen:
                continue
            seen.add(item[0])
            payloads.append(item)
    if not _has_importable(payloads):
        for item in _fallback_payloads(dist):
            if item[0] not in {arc for arc, _data in payloads}:
                payloads.append(item)
    if not payloads:
        raise VerifyOfflineError(f"No files recorded for installed distribution {name}.")
    info = Path(dist._path)
    wheel_path = f"{info.name}/WHEEL"
    wheel_text = (
        "Wheel-Version: 1.0\n"
        "Generator: llm-eval-verify-offline\n"
        "Root-Is-Purelib: true\n"
        "Tag: py3-none-any\n"
    ).encode("utf-8")
    payloads = [item for item in payloads if item[0] != wheel_path]
    payloads.append((wheel_path, wheel_text))
    return dist, payloads


def pack_distribution(name: str, dest: Path) -> Path:
    """Write one wheel for an installed distribution. No network."""
    dist, payloads = distribution_payloads(name)
    dest.mkdir(parents=True, exist_ok=True)
    filename = _wheel_filename(dist.metadata["Name"], dist.version)
    target = dest / filename
    info_name = Path(dist._path).name
    record_lines = []
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for arc, data in payloads:
            archive.writestr(arc, data)
            record_lines.append(f"{arc},{_record_hash(data)},{len(data)}")
        record_name = f"{info_name}/RECORD"
        record_lines.append(f"{record_name},,")
        archive.writestr(record_name, ("\n".join(record_lines) + "\n").encode("utf-8"))
    return target


def export_installed_wheels(dest: Path) -> list[Path]:
    written = []
    for name in closure(project_roots()):
        written.append(pack_distribution(name, dest))
    return written


def build_project_wheel(dest: Path, repo_root: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "pip",
        "wheel",
        "--no-deps",
        "--no-build-isolation",
        "--no-index",
        "-w",
        str(dest),
        str(repo_root),
    ]
    result = subprocess.run(command, cwd=str(repo_root), capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "pip wheel failed").strip()
        raise VerifyOfflineError(f"Could not build the project wheel offline. {detail[-1500:]}")
    wheels = sorted(dest.glob("llm_eval_suite-*.whl"))
    if not wheels:
        raise VerifyOfflineError("pip wheel did not write an llm_eval_suite wheel.")
    return wheels[-1]


def _venv_python(root: Path) -> Path:
    if os.name == "nt":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def install_network_guard(site: Path) -> None:
    """Install a startup hook that refuses socket connects.

    A ``sitecustomize`` module in site-packages loses to the copy in the
    standard library. A ``.pth`` import runs from this virtualenv.
    """
    site.mkdir(parents=True, exist_ok=True)
    (site / "llm_eval_netblock.py").write_text(NETBLOCK, encoding="utf-8")
    (site / "llm-eval-netblock.pth").write_text(NETBLOCK_PTH, encoding="utf-8")


def _site_packages(python: Path) -> Path:
    result = subprocess.run(
        [str(python), "-c", "import site; print(site.getsitepackages()[0])"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "site packages missing").strip()
        raise VerifyOfflineError(detail)
    return Path(result.stdout.strip())


def _clean_env(log_path: Path, extra: dict | None = None) -> dict:
    env = os.environ.copy()
    for name in PROXY_ENV:
        env.pop(name, None)
    env.pop("PYTHONPATH", None)
    env.update(OFFLINE_ENV)
    env["LLM_EVAL_NETBLOCK_LOG"] = str(log_path)
    if extra:
        env.update(extra)
    return env


def _fail_if_violations(log_path: Path, *, allow_egress: bool, what: str) -> None:
    text = log_path.read_text(encoding="utf-8") if log_path.is_file() else ""
    violations = network_violations(text, allow_egress=allow_egress)
    if violations:
        shown = "; ".join(violations[:8])
        raise VerifyOfflineError(
            f"Offline verification blocked a network attempt during {what}. Nothing was sent. {shown}"
        )


def _placeholder_detectors(dest: Path) -> None:
    from llm_eval.offline import DETECTOR_PINS

    for pin in DETECTOR_PINS:
        folder = dest / pin.key
        folder.mkdir(parents=True, exist_ok=True)
        for name in pin.files:
            (folder / name).write_bytes(b"placeholder\n")


def _placeholder_judges(dest: Path) -> None:
    from llm_eval_suite.connections import DEFAULT_JUDGES

    for row in DEFAULT_JUDGES:
        name = str(row.get("model") or "")
        safe = name.replace(":", "-").replace("/", "-")
        folder = dest / safe
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "config.json").write_text("{}\n", encoding="utf-8")


def run_installed(bundle: Path, data: Path, dataset: Path, out: Path, log_path: Path) -> None:
    """Doctor, then a stub quant-compare. Called inside the clean virtualenv."""
    from llm_eval.doctor import format_report, run_doctor
    from llm_eval.edge.cli import run_quant_compare
    from llm_eval.models.base import ModelResponse

    if not log_path.exists():
        log_path.write_text("", encoding="utf-8")
    checks = run_doctor(bundle)
    sys.stdout.write(format_report(checks))
    if not all(check.ok for check in checks):
        raise VerifyOfflineError("doctor failed. The report is above.")
    egress = next((check for check in checks if check.name == "egress"), None)
    if egress is None or not egress.ok:
        raise VerifyOfflineError(
            "A network connection succeeded. Offline verification refuses to continue."
        )

    class Stub:
        def __init__(self, profile):
            self.name = profile.get("model")

        def generate(self, prompt, **kwargs):
            return ModelResponse(text="Paris", latency_ms=1.0, tokens_used=1, metadata={})

    report = run_quant_compare(
        {
            "baseline_pair": ("fp16", "stub-fp16"),
            "variants": [("q4", "stub-q4")],
            "provider": "openai",
            "base_url": "http://127.0.0.1:9",
            "preset": "standard_capability_quick",
            "dataset": str(dataset),
            "tolerance": 0.05,
            "concurrency": 1,
            "per_suite": 1,
            "data": str(data),
            "out": str(out),
            "dry_run": False,
        },
        model_factory=lambda profile: Stub(profile),
    )
    execution = report.get("failures") or []
    if execution:
        raise VerifyOfflineError("quant-compare did not finish. " + "; ".join(str(item) for item in execution))
    print(f"quant-compare verdict: {report.get('verdict')}")
    for variant in report.get("variants") or []:
        note = str(variant.get("decision_note") or "").strip()
        if note:
            print(note)
    print(f"quant-compare report: {out / 'report.md'}")
    _fail_if_violations(log_path, allow_egress=True, what="doctor and quant-compare")


def verify(repo_root: Path | None = None) -> None:
    """Pack, install with no index, run doctor, and run the stub comparison."""
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
    fixtures = root / "standard_bench" / "fixtures"
    dataset = root / "datasets" / "sample_factcheck_50.jsonl"
    if not (fixtures / "manifest.json").is_file():
        raise VerifyOfflineError(f"Standard fixtures are missing at {fixtures}.")
    if not dataset.is_file():
        raise VerifyOfflineError(f"Sample dataset is missing at {dataset}.")

    with TemporaryDirectory(prefix="llm-eval-verify-offline-") as temp_name:
        work = Path(temp_name)
        wheels = work / "wheels"
        print("offline verify: packing wheels", flush=True)
        export_installed_wheels(wheels)
        build_project_wheel(wheels, root)
        detectors = work / "detectors"
        judges = work / "judges"
        _placeholder_detectors(detectors)
        _placeholder_judges(judges)
        bundle = work / "llm-eval-offline.tar.gz"
        print("offline verify: building bundle", flush=True)
        from llm_eval.bundle_offline import build_offline_bundle

        build_offline_bundle(
            bundle,
            repo_root=root,
            wheelhouse=wheels,
            download=False,
            platforms=["linux-x86_64"],
            python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
            data=fixtures,
            fetch=False,
            detectors=detectors,
            judges_dir=judges,
        )
        venv = work / "venv"
        print("offline verify: creating a clean venv", flush=True)
        created = subprocess.run([sys.executable, "-m", "venv", str(venv)], capture_output=True, text=True, check=False)
        if created.returncode != 0:
            detail = (created.stderr or created.stdout or "venv failed").strip()
            raise VerifyOfflineError(f"Could not create a clean virtualenv. {detail[-1500:]}")
        python = _venv_python(venv)
        site = _site_packages(python)
        install_network_guard(site)
        empty_models = work / "no-ollama"
        empty_models.mkdir()
        pip_log = work / "pip-network.log"
        pip_log.write_text("", encoding="utf-8")
        print("offline verify: pip install --no-index", flush=True)
        installed = subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--no-index",
                "--find-links",
                str(wheels),
                "llm-eval-suite[api]",
            ],
            cwd=str(work),
            env=_clean_env(pip_log, {"OLLAMA_MODELS": str(empty_models)}),
            capture_output=True,
            text=True,
            check=False,
        )
        _fail_if_violations(pip_log, allow_egress=False, what="pip install")
        if installed.returncode != 0:
            detail = (installed.stderr or installed.stdout or "pip install failed").strip()
            raise VerifyOfflineError(f"Clean install failed. {detail[-1500:]}")
        # Keep the guard in place if pip replaced site-packages contents.
        install_network_guard(site)
        run_log = work / "run-network.log"
        run_log.write_text("", encoding="utf-8")
        out = work / "quant-compare"
        print("offline verify: doctor and quant-compare", flush=True)
        child = subprocess.run(
            [
                str(python),
                "-m",
                "llm_eval",
                "verify-offline",
                "--run-installed",
                "--bundle",
                str(bundle),
                "--data",
                str(fixtures),
                "--dataset",
                str(dataset),
                "--out",
                str(out),
                "--netblock-log",
                str(run_log),
            ],
            cwd=str(work),
            env=_clean_env(run_log, {"OLLAMA_MODELS": str(empty_models)}),
            capture_output=True,
            text=True,
            check=False,
        )
        if child.stdout:
            sys.stdout.write(child.stdout)
        if child.returncode != 0:
            detail = (child.stderr or child.stdout or "installed check failed").strip()
            raise VerifyOfflineError(detail[-2000:])
        _fail_if_violations(run_log, allow_egress=True, what="doctor and quant-compare")
    print("offline verify: PASS", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="llm_eval verify-offline",
        description=(
            "Build the offline bundle, install it with pip --no-index, run doctor, "
            "and run a stub quant-compare on the standard fixtures."
        ),
    )
    parser.add_argument("--run-installed", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--bundle", default="")
    parser.add_argument("--data", default="")
    parser.add_argument("--dataset", default="")
    parser.add_argument("--out", default="")
    parser.add_argument("--netblock-log", default="")
    args = parser.parse_args(argv)
    try:
        if args.run_installed:
            run_installed(
                Path(args.bundle),
                Path(args.data),
                Path(args.dataset),
                Path(args.out),
                Path(args.netblock_log),
            )
        else:
            verify()
    except VerifyOfflineError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
