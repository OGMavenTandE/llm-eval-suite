"""Shell catalog for the click-through app.

Reserved sidebar slots stay put when a toolkit panel is not installed.
Plugin routers mount at ``/api/plugins/<name>/``. Asset files from the
panel manifest are served from ``/plugins/<name>/``.
"""

from __future__ import annotations

import json
import shutil
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.request import urlopen

from llm_eval.offline import read_settings, settings_path
from llm_eval.plugins import GROUPS, PLUGIN_API_VERSION, panel_catalog, usable

DEFAULT_TOOLKIT_ROOT = "/opt/ai-toolkit"
PACKAGE = "airgap-ai-toolkit"
RESTART_NOTE = (
    "Restart this app after you install a panel. A new entry point loads on the next start."
)
LOCAL_NOTE = "It runs on this computer only."
BANNER_COLORS = {"green", "blue", "yellow", "orange", "red", "purple"}

_TAGS = {
    "not_installed": "Not installed",
    "incompatible": "Update needed",
    "error": "Error",
    "missing": "Missing",
}

_TOOLKIT = (
    (
        "models",
        "Models",
        None,
        "List, import, run, and delete local Ollama and llama.cpp models.",
    ),
    (
        "knowledge",
        "Knowledge",
        "rag",
        "Index documents on this computer and query them with a local model.",
    ),
    (
        "finetune",
        "Fine-tune",
        "finetune",
        "Train a LoRA adapter from a local base model.",
    ),
    (
        "safety",
        "Safety",
        "safety",
        "Run local safety scans and filters before you trust a model.",
    ),
)

_CORE = (
    ("evaluate", "Evaluate", "Connect a model, run an evaluation, and read the results."),
    (
        "jobs",
        "Jobs",
        "Evaluation runs on this computer. Import, index, and training jobs show up here when those panels are installed.",
    ),
    ("settings", "Settings", "Toolkit folder, offline mode, and the classification banner."),
)


def install_command(extra: str | None, toolkit_root: str) -> str:
    wheels = str(Path(toolkit_root) / "wheels")
    package = PACKAGE if not extra else f"{PACKAGE}[{extra}]"
    return f"pip install --no-index --find-links {wheels} {package}"


def _clean_root(value: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Toolkit folder is required.")
    if "\n" in text or "\r" in text:
        raise ValueError("Toolkit folder must be one line.")
    if "://" in text or text.startswith("//"):
        raise ValueError("Toolkit folder must be a local path.")
    return text


def _clean_banner(raw: dict) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("Classification must be an object.")
    color = str(raw.get("color") or "green").strip().lower()
    if color not in BANNER_COLORS:
        raise ValueError("Classification color is not one of the banner colors.")
    text = str(raw.get("text") or "").strip()
    if len(text) > 80:
        raise ValueError("Classification text is limited to 80 characters.")
    if raw.get("enabled") and not text:
        raise ValueError("Enter the banner text.")
    return {"enabled": bool(raw.get("enabled")), "text": text, "color": color}


def shell_settings(data_dir: str | Path | None) -> dict:
    raw = read_settings(data_dir)
    root = raw.get("toolkit_root") or DEFAULT_TOOLKIT_ROOT
    banner = raw.get("classification") if isinstance(raw.get("classification"), dict) else {}
    color = str(banner.get("color") or "green").lower()
    if color not in BANNER_COLORS:
        color = "green"
    return {
        "toolkit_root": str(root),
        "classification": {
            "enabled": bool(banner.get("enabled")),
            "text": str(banner.get("text") or ""),
            "color": color,
        },
    }


def update_shell_settings(
    data_dir: str | Path,
    *,
    toolkit_root: str | None = None,
    classification: dict | None = None,
) -> dict:
    path = settings_path(data_dir)
    if path is None:
        raise ValueError("No data directory for settings.")
    payload = read_settings(data_dir)
    if toolkit_root is not None:
        payload["toolkit_root"] = _clean_root(toolkit_root)
    if classification is not None:
        payload["classification"] = _clean_banner(classification)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return shell_settings(data_dir)


def _port_open(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), 0.2):
            return True
    except OSError:
        return False


def ollama_running() -> bool:
    """True when the local Ollama tags route answers. No other host is contacted."""
    if not _port_open(11434):
        return False
    try:
        with urlopen("http://127.0.0.1:11434/api/tags", timeout=0.3) as response:
            return getattr(response, "status", 0) == 200
    except Exception:
        return False


def llama_found(toolkit_root: str) -> bool:
    root = Path(toolkit_root)
    names = ("llama-server", "llama-server.exe")
    folder = root / "llama.cpp" / "build" / "bin"
    if any((folder / name).is_file() for name in names):
        return True
    return shutil.which("llama-server") is not None


def gpu_label() -> str:
    return "GPU: none (CPU)."


def _runtime(toolkit_root: str) -> dict:
    """One probe of this computer. The footer and the toolkit list both read it."""
    ollama = ollama_running()
    llama = llama_found(toolkit_root)
    gpu = gpu_label()
    environment = " ".join(
        (
            "Ollama running." if ollama else "Ollama not running.",
            "llama.cpp found." if llama else "llama.cpp not found.",
            gpu,
        )
    )
    return {"ollama": ollama, "llama_cpp": llama, "gpu": gpu, "environment": environment}


def environment_line(toolkit_root: str) -> str:
    return _runtime(toolkit_root)["environment"]


def _package_version() -> str:
    try:
        from importlib.metadata import version

        return version("llm-eval-suite")
    except Exception:
        return "0.2.0"


def _by_slot(rows: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row.get("slot") or "more"), []).append(row)
    return grouped


def _availability(panels: list[dict], *, kind: str) -> str:
    if kind == "core":
        return "ready"
    compatible = [row for row in panels if row.get("compatible")]
    if compatible:
        primary = sorted(compatible, key=lambda row: (int(row.get("order") or 100), str(row.get("name") or "")))[0]
        return str(primary.get("availability") or "ready")
    if any(row.get("availability") == "error" for row in panels):
        return "error"
    if any(row.get("availability") == "incompatible" for row in panels):
        return "incompatible"
    if kind == "toolkit":
        return "not_installed"
    return "ready"


def _slot_row(
    slot_id: str,
    title: str,
    *,
    kind: str,
    route: str,
    summary: str,
    extra: str | None,
    toolkit_root: str,
    panels: list[dict],
) -> dict:
    availability = _availability(panels, kind=kind)
    return {
        "id": slot_id,
        "title": title,
        "kind": kind,
        "route": route,
        "availability": availability,
        "tag": _TAGS.get(availability, ""),
        "summary": summary,
        "local_note": LOCAL_NOTE if kind == "toolkit" else "",
        "command": install_command(extra, toolkit_root) if kind == "toolkit" else "",
        "extra": extra,
        "panels": [str(row.get("name") or "") for row in panels],
    }


def build_nav(rows: list[dict], toolkit_root: str) -> list[dict]:
    grouped = _by_slot(rows)
    nav = [
        _slot_row(
            "evaluate",
            "Evaluate",
            kind="core",
            route="#/evaluate",
            summary=_CORE[0][2],
            extra=None,
            toolkit_root=toolkit_root,
            panels=grouped.get("evaluate", []),
        )
    ]
    for slot_id, title, extra, summary in _TOOLKIT:
        nav.append(
            _slot_row(
                slot_id,
                title,
                kind="toolkit",
                route=f"#/{slot_id}",
                summary=summary,
                extra=extra,
                toolkit_root=toolkit_root,
                panels=grouped.get(slot_id, []),
            )
        )
    more = grouped.get("more", [])
    if more:
        nav.append(
            _slot_row(
                "more",
                "More",
                kind="more",
                route=str(more[0].get("route") or "#/more"),
                summary="Other panels installed on this computer.",
                extra=None,
                toolkit_root=toolkit_root,
                panels=more,
            )
        )
    for slot_id, title, summary in _CORE[1:]:
        nav.append(
            _slot_row(
                slot_id,
                title,
                kind="core",
                route=f"#/{slot_id}",
                summary=summary,
                extra=None,
                toolkit_root=toolkit_root,
                panels=[],
            )
        )
    return nav


_BADGES = {
    "ready": "Ready",
    "partial": "Partly ready",
    "not_installed": "Not installed",
    "incompatible": "Update needed",
    "error": "Error",
    "missing": "Missing",
}


def _needs_package(extra: str | None) -> str:
    package = PACKAGE if not extra else f"{PACKAGE}[{extra}]"
    return f"Needs {package}"


def _feature_detail(slot: dict, panels: list[dict]) -> str:
    availability = str(slot.get("availability") or "")
    if availability == "not_installed":
        return _needs_package(slot.get("extra"))
    if availability == "ready":
        return "Installed"
    if availability == "partial":
        for panel in panels:
            status = panel.get("status") if isinstance(panel.get("status"), dict) else None
            for check in (status or {}).get("checks") or []:
                if isinstance(check, dict) and not check.get("ok"):
                    text = str(check.get("detail") or check.get("fix") or "").strip()
                    if text:
                        return text
        return _needs_package(slot.get("extra"))
    for panel in panels:
        reason = str(panel.get("reason") or "").strip()
        if reason:
            return reason
    return _needs_package(slot.get("extra"))


def host_status(rows: list[dict], toolkit_root: str) -> tuple[list[dict], dict]:
    """Nav plus one status object. Footer text and toolkit rows come from that object."""
    checked_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    runtime = _runtime(toolkit_root)
    nav = build_nav(rows, toolkit_root)
    grouped = _by_slot(rows)
    features = []
    for slot in nav:
        if slot.get("kind") != "toolkit":
            continue
        availability = str(slot.get("availability") or "not_installed")
        features.append(
            {
                "id": slot["id"],
                "title": slot["title"],
                "availability": availability,
                "badge": _BADGES.get(availability, "Not installed"),
                "detail": _feature_detail(slot, grouped.get(slot["id"], [])),
            }
        )
    found = [row["title"] for row in features if row["availability"] in {"ready", "partial"}]
    status = {
        "checked_at": checked_at,
        "found": found,
        "found_label": ", ".join(found) if found else "none",
        "features": features,
        **runtime,
    }
    return nav, status


def public_catalog(rows: list[dict]) -> list[dict]:
    public = []
    for row in rows:
        item = {key: value for key, value in row.items() if not str(key).startswith("_")}
        if item.get("compatible") and item.get("has_router"):
            item["mounted_at"] = f"/api/plugins/{item['name']}"
            if not item.get("api_prefix"):
                item["api_prefix"] = item["mounted_at"]
        else:
            item["mounted_at"] = ""
        public.append(item)
    return public


def shell_payload(data_dir: str | Path | None, entry_points: Iterable | None = None) -> dict:
    rows = panel_catalog(entry_points)
    settings = shell_settings(data_dir)
    catalog = public_catalog(rows)
    nav, status = host_status(catalog, settings["toolkit_root"])
    return {
        "api_version": PLUGIN_API_VERSION,
        "groups": GROUPS,
        "panels": [row for row in catalog if row.get("compatible")],
        "catalog": catalog,
        "nav": nav,
        "checked_at": status["checked_at"],
        "status": status,
        "toolkit_root": settings["toolkit_root"],
        "classification": settings["classification"],
        "restart_note": RESTART_NOTE,
        "environment": status["environment"],
        "app_version": _package_version(),
    }


def mount_panels(app: Any, entry_points: Iterable | None = None) -> None:
    """Mount compatible panel routers and remember asset files."""
    rows = panel_catalog(entry_points)
    files: dict[str, dict[str, Path]] = {}
    for row in rows:
        if not row.get("compatible"):
            continue
        mapping: dict[str, Path] = {}
        for path in (row.get("_files") or {}).values():
            if isinstance(path, Path):
                mapping[path.name] = path
        if mapping:
            files[str(row.get("name") or "")] = mapping
    app.state.panel_files = files
    app.state.panel_entries = entry_points
    mounted: set[str] = set()
    for plugin in usable("panels", entry_points):
        if plugin.name in mounted or plugin.target is None:
            continue
        router = getattr(plugin.target, "router", None)
        if router is None:
            continue
        app.include_router(router, prefix=f"/api/plugins/{plugin.name}")
        mounted.add(plugin.name)
