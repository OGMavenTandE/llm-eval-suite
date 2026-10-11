"""SHA-256 manifest for pinned local benchmark files."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


class HashMismatch(RuntimeError):
    """The local file does not match the manifest. The run stops."""


class ManifestError(RuntimeError):
    """The manifest is missing or does not describe this suite."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest_path(dest: str | Path) -> Path:
    return Path(dest) / "manifest.json"


def load_manifest(dest: str | Path) -> dict:
    path = manifest_path(dest)
    if not path.is_file():
        raise ManifestError(
            f"No manifest at {path}. Run python -m standard_bench fetch --dest {dest} "
            "on a connected machine, then copy the directory. Nothing was downloaded."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Manifest at {path} is not valid JSON.") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("suites"), dict):
        raise ManifestError(f"Manifest at {path} has no suites object.")
    return payload


def write_manifest(dest: str | Path, payload: dict) -> Path:
    path = manifest_path(dest)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def file_record(dest: Path, relative: str) -> dict:
    path = dest / relative
    data = path.read_bytes()
    return {
        "path": relative.replace("\\", "/"),
        "sha256": sha256_bytes(data),
        "bytes": len(data),
    }


def verify_suite(dest: str | Path, suite: str) -> dict:
    """Raise HashMismatch or ManifestError before a run touches the rows."""
    root = Path(dest)
    payload = load_manifest(root)
    entry = (payload.get("suites") or {}).get(suite)
    if not isinstance(entry, dict):
        raise ManifestError(
            f"Manifest at {manifest_path(root)} has no entry for {suite}. "
            "Fetch that suite before running it."
        )
    files = entry.get("files") or []
    if not files:
        raise ManifestError(f"Manifest entry for {suite} lists no files.")
    for record in files:
        relative = str(record.get("path") or "")
        expected = str(record.get("sha256") or "")
        path = root / relative
        if not path.is_file():
            raise HashMismatch(
                f"{suite} file is missing: {path}. The run will not start. "
                "Refetch the suite on a connected machine."
            )
        actual = sha256_file(path)
        if actual != expected:
            raise HashMismatch(
                f"{suite} hash mismatch for {relative}. "
                f"manifest {expected} local {actual}. "
                "The run will not start. Refetch the suite. Nothing was scored."
            )
    return entry
