"""One tarball for a closed network.

``python -m llm_eval bundle-offline`` packs a Linux x86_64 wheelhouse, a
Windows wheelhouse when those wheels can be downloaded, the standard
capability kit, garak detector files when they are already on disk, and
either the local judge files or a manifest that names them.

Judge weights stay out unless ``--judges-dir`` is set. The two default
council judges are about 2 GB each.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from llm_eval.offline import DETECTOR_PINS
from llm_eval_suite.eval_env import (
    CUDA_INDEX,
    GARAK_VERSION,
    TORCH_VERSION,
    TRANSFORMERS_VERSION,
)

BUNDLE_DIRNAME = "llm-eval-offline"
SHA_NAME = "SHA256SUMS"
KIT_NAME = "standard-capability-kit.zip"
PLATFORM_PIP = {
    "linux-x86_64": "manylinux2014_x86_64",
    "win-amd64": "win_amd64",
}
PYTHON_ABIS = {"3.10": "cp310", "3.11": "cp311", "3.12": "cp312"}
API_DOWNLOADS = [
    "pydantic>=2.0",
    "pyyaml>=6.0",
    "requests>=2.28.0",
    "fastapi>=0.100",
    "uvicorn[standard]>=0.23",
    "httpx>=0.24",
    "setuptools>=68",
    "wheel",
]
DETECTOR_SIZE_NOTES = {
    "refutation_detector_distilbert": "about 260 MB",
    "roberta_nli": "about 1.4 GB",
}
JUDGE_SIZE_NOTES = {
    "qwen2.5:3b-instruct": "about 2 GB as an Ollama tag",
    "llama3.2:3b": "about 2 GB as an Ollama tag",
}
JUDGE_TRADEOFF = (
    "qwen2.5:3b-instruct and llama3.2:3b are about 2 GB each, about 4 GB together. "
    "They are left out of the tarball unless --judges-dir is set, so the bundle stays "
    "a wheelhouse plus the capability kit. On the closed network, pass that same "
    "directory to python -m llm_eval doctor --judges-dir."
)

INSTALL_SH = """#!/bin/sh
# Offline install. This script does not use the network.
set -eu
ROOT=$(CDPATH= cd -- "$(dirname "$0")" && pwd)
cd "$ROOT"

if command -v python3 >/dev/null 2>&1; then
  PY=python3
else
  PY=python
fi

"$PY" -c 'import json,sys
m=json.load(open(sys.argv[1], encoding="utf-8"))
need=tuple(int(p) for p in str(m["python"]).split(".")[:2])
sys.exit(0 if sys.version_info[:2]==need else 1)' "$ROOT/manifest.json" || {
  echo "FAIL  python  install requires the Python version in manifest.json" >&2
  exit 1
}

WHEELHOUSE="$ROOT/wheelhouse/linux-x86_64"
COMMON="$ROOT/wheelhouse/common"
if [ ! -d "$WHEELHOUSE" ] && [ ! -d "$COMMON" ]; then
  echo "FAIL  wheelhouse  linux-x86_64 wheelhouse is missing" >&2
  exit 1
fi

KIT=""
for candidate in "$ROOT/kit"/*.zip; do
  if [ -f "$candidate" ]; then
    KIT=$candidate
    break
  fi
done
if [ -z "$KIT" ]; then
  echo "FAIL  kit  no zip under kit/" >&2
  exit 1
fi

"$PY" -m venv "$ROOT/.venv-eval"
# shellcheck disable=SC1091
. "$ROOT/.venv-eval/bin/activate"
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INDEX=1
export LLM_EVAL_OFFLINE=1
export LLM_EVAL_AIRGAP_BUILD=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export LLM_EVAL_GARAK_DETECTORS_DIR="$ROOT/garak_detectors"

set -- --no-index --no-build-isolation
if [ -d "$WHEELHOUSE" ]; then
  set -- "$@" --find-links "$WHEELHOUSE"
fi
if [ -d "$COMMON" ]; then
  set -- "$@" --find-links "$COMMON"
fi

python -m pip install "$@" "setuptools" "wheel" || true
python -m pip install "$@" "torch==__TORCH__" "garak==__GARAK__" "transformers==__TRANSFORMERS__"

SRC="$ROOT/src"
rm -rf "$SRC"
mkdir -p "$SRC"
python -c 'import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])' "$KIT" "$SRC"
python -m pip install "$@" "$SRC[api]"

python -m llm_eval doctor --bundle "$ROOT"
"""


class BundleOfflineError(RuntimeError):
    """The bundle could not be built. Nothing useful was written."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def host_platform_tag() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "linux" and machine in {"x86_64", "amd64"}:
        return "linux-x86_64"
    if system == "windows" and machine in {"x86_64", "amd64"}:
        return "win-amd64"
    return f"{system}-{machine}"


def render_install_sh() -> str:
    return (
        INSTALL_SH.replace("__TORCH__", TORCH_VERSION)
        .replace("__GARAK__", GARAK_VERSION)
        .replace("__TRANSFORMERS__", TRANSFORMERS_VERSION)
    )


def _pip_base(dest: Path, python_version: str, platform_name: str) -> list[str]:
    abi = PYTHON_ABIS.get(python_version)
    pip_platform = PLATFORM_PIP.get(platform_name)
    if abi is None or pip_platform is None:
        raise BundleOfflineError(
            f"Unsupported platform or Python for a wheelhouse: {platform_name} python {python_version}."
        )
    pyver = python_version.replace(".", "")
    return [
        sys.executable,
        "-m",
        "pip",
        "download",
        "-d",
        str(dest),
        "--python-version",
        pyver,
        "--platform",
        pip_platform,
        "--implementation",
        "cp",
        "--abi",
        abi,
        "--only-binary=:all:",
    ]


def host_linux_commands(dest: Path) -> list[list[str]]:
    """Wheelhouse commands for a Linux x86_64 builder matching this interpreter."""
    torch = f"torch=={TORCH_VERSION}"
    garak = f"garak=={GARAK_VERSION}"
    transformers = f"transformers=={TRANSFORMERS_VERSION}"
    return [
        [sys.executable, "-m", "pip", "download", "-d", str(dest), torch, "--index-url", CUDA_INDEX],
        [
            sys.executable,
            "-m",
            "pip",
            "download",
            "-d",
            str(dest),
            garak,
            transformers,
            torch,
            "--extra-index-url",
            CUDA_INDEX,
        ],
        [sys.executable, "-m", "pip", "download", "-d", str(dest), ".[api]"],
        [sys.executable, "-m", "pip", "download", "-d", str(dest), "setuptools>=68", "wheel"],
    ]


def constrained_commands(dest: Path, platform_name: str, python_version: str) -> list[list[str]]:
    """Cross-download. Windows is best-effort. The local project is not a binary wheel."""
    base = _pip_base(dest, python_version, platform_name)
    torch = f"torch=={TORCH_VERSION}"
    garak = f"garak=={GARAK_VERSION}"
    transformers = f"transformers=={TRANSFORMERS_VERSION}"
    return [
        [*base, torch, "--index-url", CUDA_INDEX],
        [*base, garak, transformers, torch, "--extra-index-url", CUDA_INDEX],
        [*base, *API_DOWNLOADS],
    ]


def linux_download_plan(dest: Path, python_version: str) -> tuple[str, list[list[str]]]:
    running = f"{sys.version_info.major}.{sys.version_info.minor}"
    if host_platform_tag() == "linux-x86_64" and python_version == running:
        return "host", host_linux_commands(dest)
    return "constrained", constrained_commands(dest, "linux-x86_64", python_version)


def _run_commands(commands: list[list[str]], cwd: Path, runner) -> subprocess.CompletedProcess:
    last = subprocess.CompletedProcess(commands[0] if commands else [], 0, "", "")
    for command in commands:
        last = runner(command, cwd=str(cwd), check=False, capture_output=True, text=True)
        if last.returncode != 0:
            return last
    return last


def _package_count(dest: Path) -> int:
    if not dest.is_dir():
        return 0
    return sum(1 for path in dest.iterdir() if path.is_file() and path.suffix in {".whl", ".gz", ".zip"})


def download_wheelhouses(
    wheel_root: Path,
    platforms: list[str],
    python_version: str,
    repo_root: Path,
    runner=None,
) -> dict:
    """Download wheels. A Windows failure is recorded. A Linux failure stops the bundle."""
    if "linux-x86_64" not in platforms:
        raise BundleOfflineError("The bundle always needs a linux-x86_64 wheelhouse.")
    runner = runner or subprocess.run
    reports: dict[str, dict] = {}
    linux_dest = wheel_root / "linux-x86_64"
    linux_dest.mkdir(parents=True, exist_ok=True)
    mode, commands = linux_download_plan(linux_dest, python_version)
    result = _run_commands(commands, repo_root, runner)
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "pip download failed").strip()
        raise BundleOfflineError(f"Linux wheelhouse download failed. {detail[-1500:]}")
    reports["linux-x86_64"] = {
        "present": True,
        "mode": mode,
        "packages": _package_count(linux_dest),
        "python": python_version,
    }
    if "win-amd64" in platforms:
        win_dest = wheel_root / "win-amd64"
        win_dest.mkdir(parents=True, exist_ok=True)
        win_commands = constrained_commands(win_dest, "win-amd64", python_version)
        win_result = _run_commands(win_commands, repo_root, runner)
        if win_result.returncode != 0:
            shutil.rmtree(win_dest, ignore_errors=True)
            detail = (win_result.stderr or win_result.stdout or "pip download failed").strip()
            reports["win-amd64"] = {
                "present": False,
                "mode": "constrained",
                "packages": 0,
                "python": python_version,
                "reason": detail[-1500:] or "Windows wheels were not available from this machine.",
            }
        else:
            reports["win-amd64"] = {
                "present": True,
                "mode": "constrained",
                "packages": _package_count(win_dest),
                "python": python_version,
            }
    return reports


def _platform_for_filename(name: str, default: str) -> str:
    lowered = name.lower()
    if "win_amd64" in lowered or lowered.find("win32") >= 0:
        return "win-amd64"
    if "manylinux" in lowered or "linux_x86_64" in lowered or "musllinux" in lowered:
        return "linux-x86_64"
    if lowered.endswith(".whl") and "none-any" in lowered:
        return "common"
    return default


def stage_supplied_wheelhouse(source: Path, dest_root: Path) -> dict:
    if not source.is_dir():
        raise BundleOfflineError(f"No wheelhouse directory at {source}.")
    known = {"linux-x86_64", "win-amd64", "common"}
    subdirs = [path for path in source.iterdir() if path.is_dir() and path.name in known]
    files = [path for path in source.iterdir() if path.is_file() and path.suffix in {".whl", ".gz", ".zip"}]
    if subdirs:
        for sub in subdirs:
            _copy_tree(sub, dest_root / sub.name)
    elif files:
        default = host_platform_tag() if host_platform_tag() in PLATFORM_PIP else "linux-x86_64"
        for path in files:
            bucket = _platform_for_filename(path.name, default)
            target = dest_root / bucket
            target.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target / path.name)
    else:
        raise BundleOfflineError(f"No wheels in {source}.")
    reports = {}
    for name in ("linux-x86_64", "win-amd64", "common"):
        folder = dest_root / name
        count = _package_count(folder)
        if count:
            reports[name] = {"present": True, "mode": "prebuilt", "packages": count}
    if not reports.get("linux-x86_64", {}).get("present") and not reports.get("common", {}).get("present"):
        raise BundleOfflineError(
            "The wheelhouse has no linux-x86_64 or pure-Python wheels. "
            "install.sh cannot install from it."
        )
    if "win-amd64" not in reports:
        reports["win-amd64"] = {
            "present": False,
            "mode": "prebuilt",
            "packages": 0,
            "reason": "The supplied wheelhouse has no Windows wheels.",
        }
    return reports


def _copy_tree(source: Path, dest: Path) -> int:
    count = 0
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if any(part in {"__pycache__", ".git"} for part in relative.parts):
            continue
        target = dest / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        count += 1
    return count


def _resolve_kit(
    kit_dir: Path,
    *,
    repo_root: Path,
    kit: Path | None,
    data: Path | None,
    fetch: bool,
    allow_moved: bool,
) -> dict:
    from standard_bench.bundle import BundleError, build_bundle
    from standard_bench.manifest import HashMismatch, ManifestError

    destination = kit_dir / KIT_NAME
    kit_dir.mkdir(parents=True, exist_ok=True)
    try:
        if kit is not None:
            if not kit.is_file():
                raise BundleOfflineError(f"No kit zip at {kit}.")
            shutil.copy2(kit, destination)
            return {"path": f"kit/{KIT_NAME}", "bytes": destination.stat().st_size, "source": "copy"}
        data_dir = data
        if data_dir is None:
            local = repo_root / "datasets" / "standard"
            if (local / "manifest.json").is_file():
                data_dir = local
        if data_dir is not None:
            info = build_bundle(destination, data_dir, repo_root=repo_root)
            return {"path": f"kit/{KIT_NAME}", "bytes": info["bytes"], "source": "local-data"}
        if not fetch:
            raise BundleOfflineError(
                "No standard capability kit. Pass --kit or --data, or omit --no-fetch on a connected machine."
            )
        from standard_bench.fetch import FetchError, fetch as fetch_suites
        from standard_bench.protocol import CUSTOMER_SUITES

        with TemporaryDirectory(prefix="standard-kit-") as temp:
            fetch_suites(temp, list(CUSTOMER_SUITES), allow_moved=allow_moved)
            info = build_bundle(destination, temp, repo_root=repo_root)
        return {"path": f"kit/{KIT_NAME}", "bytes": info["bytes"], "source": "fetch"}
    except (BundleError, ManifestError, HashMismatch) as exc:
        raise BundleOfflineError(str(exc)) from exc


def _stage_attributions(staging: Path, repo_root: Path) -> None:
    """Copy vendored MPL-2.0 texts and the Llama attribution into the tarball."""
    notice = repo_root / "NOTICE"
    if not notice.is_file():
        raise BundleOfflineError("NOTICE is missing. It must say Built with Llama.")
    text = notice.read_text(encoding="utf-8")
    if "Built with Llama" not in text:
        raise BundleOfflineError("NOTICE must include the Built with Llama attribution.")
    shutil.copy2(notice, staging / "NOTICE")
    source = repo_root / "third_party"
    required = (
        source / "certifi" / "LICENSE",
        source / "certifi" / "MPL-2.0",
        source / "tqdm" / "LICENSE",
        source / "tqdm" / "MPL-2.0",
    )
    missing = [str(path.relative_to(repo_root)) for path in required if not path.is_file()]
    if missing:
        raise BundleOfflineError("Missing third-party license files: " + ", ".join(missing) + ".")
    for path in required:
        mpl = path.read_text(encoding="utf-8")
        if "Mozilla Public License" not in mpl and "MPL" not in mpl:
            raise BundleOfflineError(f"{path.name} under {path.parent.name} is not an MPL-2.0 license file.")
    _copy_tree(source, staging / "third_party")


def _stage_detectors(staging: Path, source: Path | None) -> list[dict]:
    root = staging / "garak_detectors"
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for pin in DETECTOR_PINS:
        folder = (source / pin.key) if source is not None else pin.directory()
        missing = [name for name in pin.files if not (folder / name).is_file()]
        included = not missing and folder.is_dir()
        if included:
            _copy_tree(folder, root / pin.key)
        rows.append(
            {
                "key": pin.key,
                "repo_id": pin.repo_id,
                "revision": pin.revision,
                "files": list(pin.files),
                "included": included,
                "missing": missing,
                "approx": DETECTOR_SIZE_NOTES.get(pin.key, ""),
            }
        )
    payload = {
        "version": 1,
        "note": (
            "Government T&E loads these two garak detectors. "
            "They are copied only when every listed file is already on the build machine. "
            "Nothing is downloaded for them here."
        ),
        "detectors": rows,
    }
    (root / "DETECTORS.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return rows


def _judge_rows(suite_data: Path | None) -> list[dict]:
    from llm_eval_suite.connections import DEFAULT_JUDGES

    if suite_data is not None and (suite_data / "judges.json").is_file():
        try:
            payload = json.loads((suite_data / "judges.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise BundleOfflineError(f"judges.json in {suite_data} is not valid JSON.") from exc
        rows = payload.get("judges") if isinstance(payload, dict) else None
        if isinstance(rows, list) and rows:
            return rows
    return list(DEFAULT_JUDGES)


def _stage_judges(staging: Path, judges_dir: Path | None, models: list[dict]) -> dict:
    root = staging / "judges"
    root.mkdir(parents=True, exist_ok=True)
    copied = 0
    byte_count = 0
    if judges_dir is not None:
        if not judges_dir.is_dir():
            raise BundleOfflineError(f"No judge directory at {judges_dir}.")
        copied = _copy_tree(judges_dir, root)
        byte_count = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    described = []
    for row in models:
        name = str(row.get("model") or row.get("name") or "").strip()
        if not name:
            continue
        described.append(
            {
                "name": name,
                "type": row.get("type") or "ollama",
                "approx": JUDGE_SIZE_NOTES.get(name, "size depends on the quant and the format"),
            }
        )
    payload = {
        "version": 1,
        "included": copied > 0,
        "bytes": byte_count,
        "tradeoff": JUDGE_TRADEOFF,
        "models": described,
    }
    (root / "JUDGES.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def _write_manifest(staging: Path, body: dict) -> None:
    files = []
    for path in sorted(staging.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(staging).as_posix()
        if relative == "manifest.json":
            continue
        files.append({"path": relative, "sha256": file_sha256(path), "bytes": path.stat().st_size})
    body = dict(body)
    body["files"] = files
    (staging / "manifest.json").write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")


def write_sha256sums(staging: Path) -> int:
    lines = []
    for path in sorted(staging.rglob("*")):
        if not path.is_file() or path.name == SHA_NAME:
            continue
        relative = path.relative_to(staging).as_posix()
        lines.append(f"{file_sha256(path)}  {relative}")
    text = "\n".join(lines) + "\n"
    (staging / SHA_NAME).write_text(text, encoding="utf-8", newline="\n")
    return len(lines)


def verify_sha256sums(root: Path) -> tuple[bool, str]:
    sums = root / SHA_NAME
    if not sums.is_file():
        return False, "SHA256SUMS is missing"
    mismatches: list[str] = []
    checked = 0
    for raw in sums.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "  " in line:
            digest, relative = line.split("  ", 1)
        else:
            parts = line.split(None, 1)
            if len(parts) != 2:
                mismatches.append("unreadable line")
                continue
            digest, relative = parts
        relative = relative.strip().lstrip("*")
        checked += 1
        target = root / relative
        if not target.is_file():
            mismatches.append(f"{relative} missing")
            continue
        if file_sha256(target) != digest:
            mismatches.append(relative)
    if mismatches:
        shown = ", ".join(mismatches[:5])
        return False, f"{len(mismatches)} mismatch: {shown}"
    if checked == 0:
        return False, "SHA256SUMS is empty"
    return True, f"{checked} files"


def _tar_filter(info: tarfile.TarInfo) -> tarfile.TarInfo:
    if info.name.endswith("/install.sh"):
        info.mode = 0o755
    return info


def build_offline_bundle(
    output: str | Path,
    *,
    repo_root: str | Path | None = None,
    wheelhouse: str | Path | None = None,
    download: bool = True,
    platforms: list[str] | None = None,
    python_version: str = "3.11",
    kit: str | Path | None = None,
    data: str | Path | None = None,
    fetch: bool = True,
    allow_moved: bool = False,
    detectors: str | Path | None = None,
    judges_dir: str | Path | None = None,
    suite_data: str | Path | None = None,
    runner=None,
) -> dict:
    """Stage the bundle and write one gzipped tarball. Returns a short summary."""
    root = Path(repo_root) if repo_root else Path(__file__).resolve().parents[1]
    destination = Path(output)
    chosen = list(platforms or ("linux-x86_64", "win-amd64"))
    unknown = [name for name in chosen if name not in PLATFORM_PIP]
    if unknown:
        raise BundleOfflineError("Unknown platform: " + ", ".join(unknown) + ".")
    wheel_source = Path(wheelhouse) if wheelhouse else None
    if wheel_source is None and not download:
        raise BundleOfflineError("Pass --wheelhouse, or omit --no-download so wheels can be fetched.")

    with TemporaryDirectory(prefix="llm-eval-offline-") as temp_name:
        staging = Path(temp_name) / "stage"
        staging.mkdir()
        wheel_root = staging / "wheelhouse"
        wheel_root.mkdir()
        if wheel_source is not None:
            platform_report = stage_supplied_wheelhouse(wheel_source, wheel_root)
        else:
            platform_report = download_wheelhouses(wheel_root, chosen, python_version, root, runner=runner)
        kit_info = _resolve_kit(
            staging / "kit",
            repo_root=root,
            kit=Path(kit) if kit else None,
            data=Path(data) if data else None,
            fetch=fetch,
            allow_moved=allow_moved,
        )
        detector_source = Path(detectors) if detectors else None
        detector_rows = _stage_detectors(staging, detector_source)
        judge_payload = _stage_judges(
            staging,
            Path(judges_dir) if judges_dir else None,
            _judge_rows(Path(suite_data) if suite_data else None),
        )
        script = staging / "install.sh"
        script.write_text(render_install_sh(), encoding="utf-8", newline="\n")
        script.chmod(0o755)
        _stage_attributions(staging, root)
        manifest = {
            "version": 1,
            "kind": "llm-eval-offline-bundle",
            "created": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "python": python_version,
            "pins": {
                "torch": TORCH_VERSION,
                "garak": GARAK_VERSION,
                "transformers": TRANSFORMERS_VERSION,
            },
            "platforms": platform_report,
            "kit": kit_info,
            "detectors": detector_rows,
            "judges": {
                "included": judge_payload["included"],
                "bytes": judge_payload["bytes"],
                "tradeoff": judge_payload["tradeoff"],
                "models": [row["name"] for row in judge_payload["models"]],
            },
        }
        _write_manifest(staging, manifest)
        file_count = write_sha256sums(staging)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(destination, "w:gz") as archive:
            archive.add(staging, arcname=BUNDLE_DIRNAME, filter=_tar_filter)
    size = destination.stat().st_size
    return {
        "path": destination,
        "bytes": size,
        "megabytes": size / (1024 * 1024),
        "file_count": file_count,
        "platforms": platform_report,
        "kit": kit_info,
        "detectors_included": sum(1 for row in detector_rows if row["included"]),
        "judges_included": bool(judge_payload["included"]),
        "judge_bytes": judge_payload["bytes"],
    }


def _print_summary(summary: dict) -> None:
    path = summary["path"]
    print(f"{path} {summary['bytes']} bytes ({summary['megabytes']:.2f} MB)")
    for name, report in summary["platforms"].items():
        if report.get("present"):
            print(f"wheels {name} {report.get('packages', 0)} packages")
        else:
            print(f"wheels {name} not included")
    print(f"kit {summary['kit']['bytes']} bytes")
    print(f"detectors included {summary['detectors_included']}")
    if summary["judges_included"]:
        print(f"judges included {summary['judge_bytes']} bytes")
    else:
        print("judges not included (manifest only)")
        print(JUDGE_TRADEOFF)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm_eval bundle-offline",
        description="Pack a wheelhouse, the standard capability kit, and offline resources into one tarball.",
    )
    parser.add_argument("--output", default="dist/llm-eval-offline.tar.gz")
    parser.add_argument("--wheelhouse", default="", help="Directory of wheels already downloaded. Skips pip download.")
    parser.add_argument("--no-download", action="store_true", help="Do not contact a package index.")
    parser.add_argument("--platforms", default="linux-x86_64,win-amd64")
    parser.add_argument("--python-version", default="3.11")
    parser.add_argument("--kit", default="", help="Existing standard capability kit zip.")
    parser.add_argument("--data", default="", help="Local dataset directory to pack. Skips a fetch.")
    parser.add_argument("--no-fetch", action="store_true", help="Do not download the capability datasets.")
    parser.add_argument("--allow-moved", action="store_true")
    parser.add_argument("--detectors", default="", help="Parent directory of the garak detector folders.")
    parser.add_argument(
        "--judges-dir",
        default="",
        help="Copy this judge model directory into the tarball. Omit it to write a manifest only.",
    )
    parser.add_argument(
        "--suite-data",
        default="",
        help="App data directory whose judges.json names the council. Default judges are used when it is omitted.",
    )
    parser.add_argument("--repo-root", default="")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    platforms = [part.strip() for part in args.platforms.split(",") if part.strip()]
    try:
        summary = build_offline_bundle(
            args.output,
            repo_root=args.repo_root or None,
            wheelhouse=args.wheelhouse or None,
            download=not args.no_download,
            platforms=platforms,
            python_version=args.python_version,
            kit=args.kit or None,
            data=args.data or None,
            fetch=not args.no_fetch,
            allow_moved=bool(args.allow_moved),
            detectors=args.detectors or None,
            judges_dir=args.judges_dir or None,
            suite_data=args.suite_data or None,
        )
    except BundleOfflineError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    _print_summary(summary)
    raise SystemExit(0)
