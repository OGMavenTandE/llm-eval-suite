"""One zip of the test kit and the six customer datasets.

GPQA Diamond is never packed. Model weights are not packed. The zip carries
a SHA-256 manifest, and a run from an unpacked kit checks those hashes.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from standard_bench.manifest import HashMismatch, ManifestError, load_manifest, sha256_bytes, verify_suite
from standard_bench.protocol import CUSTOMER_SUITES, GPQA_SUITE

KIT_MANIFEST = "kit-manifest.json"
DATA_ROOT = "datasets/standard"
MAX_KIT_BYTES = 35 * 1024 * 1024

INCLUDE_DIRS = ("standard_bench", "llm_eval", "llm_eval_suite", "dow_bench")
INCLUDE_FILES = ("pyproject.toml", "README.md")
INCLUDE_DOCS = (
    "docs/AIRGAP_INSTALL.md",
    "docs/STANDARD_BENCH.md",
    "docs/STANDARD_BENCH_RUN_PLAN.md",
    "docs/LICENSES.md",
)
SKIP_PARTS = {
    "__pycache__",
    ".pytest_cache",
    ".git",
    "gpqa_diamond",
    "node_modules",
}
SKIP_SUFFIXES = {".pyc", ".pyo", ".pt", ".safetensors", ".bin", ".gguf"}
# In-house instructions stay in the repo. They are not part of the customer zip.
SKIP_FILES = {"docs/GPQA_INHOUSE.md"}


class BundleError(RuntimeError):
    pass


def _repo_root(explicit: str | Path | None) -> Path:
    if explicit:
        return Path(explicit)
    return Path(__file__).resolve().parents[1]


def _skipped(relative: str) -> bool:
    path = Path(relative)
    if relative.replace("\\", "/") in SKIP_FILES:
        return True
    if any(part in SKIP_PARTS for part in path.parts):
        return True
    if path.suffix.lower() in SKIP_SUFFIXES:
        return True
    if GPQA_SUITE in path.parts:
        return True
    return False


def _strip_gpqa_manifest(data: bytes) -> bytes:
    try:
        payload = json.loads(data.decode("utf-8"))
    except json.JSONDecodeError:
        return data
    suites = payload.get("suites") if isinstance(payload, dict) else None
    if not isinstance(suites, dict) or GPQA_SUITE not in suites:
        return data
    payload["suites"] = {key: value for key, value in suites.items() if key != GPQA_SUITE}
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _code_files(root: Path) -> list[tuple[str, bytes]]:
    found: list[tuple[str, bytes]] = []
    for name in INCLUDE_FILES:
        path = root / name
        if path.is_file() and not _skipped(name):
            found.append((name, path.read_bytes()))
    for name in INCLUDE_DOCS:
        path = root / name
        if path.is_file() and not _skipped(name):
            found.append((name.replace("\\", "/"), path.read_bytes()))
    for directory in INCLUDE_DIRS:
        base = root / directory
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            if _skipped(relative):
                continue
            data = path.read_bytes()
            if path.name == "manifest.json":
                data = _strip_gpqa_manifest(data)
            found.append((relative, data))
    return found


def _data_files(data: Path) -> list[tuple[str, bytes]]:
    try:
        payload = load_manifest(data)
    except ManifestError as exc:
        raise BundleError(str(exc)) from exc
    suites = {}
    files: list[tuple[str, bytes]] = []
    for name in CUSTOMER_SUITES:
        entry = (payload.get("suites") or {}).get(name)
        if not isinstance(entry, dict):
            raise BundleError(
                f"Customer kit needs {name} in {data}. "
                "GPQA is not part of the kit. Nothing was packed."
            )
        verify_suite(data, name)
        cleaned_files = []
        for record in entry.get("files") or []:
            relative = str(record.get("path") or "").replace("\\", "/")
            if GPQA_SUITE in Path(relative).parts or "gpqa" in relative.lower():
                raise BundleError(f"Refusing to pack a GPQA path: {relative}.")
            blob = (data / relative).read_bytes()
            files.append((f"{DATA_ROOT}/{relative}", blob))
            cleaned_files.append(record)
        suites[name] = {key: value for key, value in entry.items() if key != "files"}
        suites[name]["files"] = cleaned_files
        suites[name]["gated"] = False
    manifest = {"version": 1, "suites": suites}
    files.append((f"{DATA_ROOT}/manifest.json", (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")))
    return files


def _is_dataset_payload(relative: str) -> bool:
    path = relative.replace("\\", "/")
    return path.endswith(".jsonl") or path == f"{DATA_ROOT}/manifest.json" or path.endswith("/fixtures/manifest.json")


def _reject_gpqa(entries: list[tuple[str, bytes]]) -> None:
    for relative, data in entries:
        path = relative.replace("\\", "/")
        if GPQA_SUITE in Path(path).parts:
            raise BundleError(f"Refusing to pack GPQA data at {path}.")
        if _is_dataset_payload(path) and (b"GPQA_SENTINEL_QUESTION" in data or b"gpqa_diamond" in data):
            raise BundleError(f"Refusing to pack GPQA data in {path}.")


def _kit_manifest(entries: list[tuple[str, bytes]]) -> bytes:
    files = []
    for relative, data in entries:
        files.append(
            {
                "path": relative.replace("\\", "/"),
                "sha256": sha256_bytes(data),
                "bytes": len(data),
            }
        )
    payload = {
        "version": 1,
        "kind": "standard-capability-kit",
        "suites": list(CUSTOMER_SUITES),
        "weights_included": False,
        "files": files,
    }
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def build_bundle(
    output: str | Path,
    data: str | Path,
    repo_root: str | Path | None = None,
) -> dict:
    """Pack the test kit source and the six customer datasets. No network."""
    root = _repo_root(repo_root)
    data_dir = Path(data)
    if not data_dir.is_dir():
        raise BundleError(f"No dataset directory at {data_dir}.")
    entries = _code_files(root) + _data_files(data_dir)
    _reject_gpqa(entries)
    manifest = _kit_manifest(entries)
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative, blob in entries:
            archive.writestr(relative, blob)
        archive.writestr(KIT_MANIFEST, manifest)
    size = destination.stat().st_size
    return {
        "path": destination,
        "bytes": size,
        "megabytes": size / (1024 * 1024),
        "file_count": len(entries) + 1,
        "over_budget": size >= MAX_KIT_BYTES,
    }


def _members_from_zip(path: Path) -> dict[str, bytes]:
    with zipfile.ZipFile(path) as archive:
        return {info.filename.replace("\\", "/"): archive.read(info) for info in archive.infolist() if not info.is_dir()}


def _members_from_dir(path: Path) -> dict[str, bytes]:
    found = {}
    for child in path.rglob("*"):
        if child.is_file():
            found[child.relative_to(path).as_posix()] = child.read_bytes()
    return found


def verify_kit(root: str | Path) -> dict:
    """Check the kit manifest. ``root`` may be a zip or an unpacked directory."""
    path = Path(root)
    if path.is_file() and path.suffix.lower() == ".zip":
        members = _members_from_zip(path)
        base = None
    elif path.is_dir():
        members = _members_from_dir(path)
        base = path
    else:
        raise ManifestError(f"No kit at {path}.")
    raw = members.get(KIT_MANIFEST)
    if raw is None and base is not None:
        raise ManifestError(f"No {KIT_MANIFEST} in {path}.")
    if raw is None:
        raise ManifestError(f"No {KIT_MANIFEST} in {path}.")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"{KIT_MANIFEST} is not valid JSON.") from exc
    suites = payload.get("suites") or []
    if GPQA_SUITE in suites:
        raise BundleError("Kit manifest lists GPQA. The customer kit must not.")
    for record in payload.get("files") or []:
        relative = str(record.get("path") or "").replace("\\", "/")
        if GPQA_SUITE in Path(relative).parts:
            raise BundleError(f"Kit manifest lists GPQA data at {relative}.")
        expected = str(record.get("sha256") or "")
        blob = members.get(relative)
        if blob is None:
            raise HashMismatch(f"Kit file is missing: {relative}.")
        actual = sha256_bytes(blob)
        if actual != expected:
            raise HashMismatch(
                f"Kit hash mismatch for {relative}. manifest {expected} local {actual}."
            )
        if _is_dataset_payload(relative) and (b"GPQA_SENTINEL_QUESTION" in blob or b"gpqa_diamond" in blob):
            raise BundleError(f"Kit file contains GPQA data: {relative}.")
    data_manifest = members.get(f"{DATA_ROOT}/manifest.json")
    if data_manifest is not None:
        data_payload = json.loads(data_manifest.decode("utf-8"))
        names = set((data_payload.get("suites") or {}))
        if GPQA_SUITE in names:
            raise BundleError("Dataset manifest inside the kit lists GPQA.")
        missing = [name for name in CUSTOMER_SUITES if name not in names]
        if missing:
            raise ManifestError("Kit dataset manifest is missing: " + ", ".join(missing) + ".")
    return payload


def verify_kit_near(data_dir: Path) -> None:
    """If this dataset sits inside an unpacked kit, check the kit hashes first."""
    current = Path(data_dir).resolve()
    for _ in range(4):
        candidate = current / KIT_MANIFEST
        if candidate.is_file():
            verify_kit(current)
            return
        if current.parent == current:
            return
        current = current.parent

