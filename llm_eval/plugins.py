"""Plugin API 0.2.0.

Third-party packages register entry points in these groups:

- ``llm_eval.suites``
- ``llm_eval.scorers``
- ``llm_eval.reporters``
- ``llm_eval.connectors``
- ``llm_eval.installers``
- ``llm_eval.panels``

Each object exposes ``name``, ``version``, and ``api_version``. The host
loads a plugin when the major number of ``api_version`` matches
``PLUGIN_API_VERSION``. A run manifest records the plugins that ran.

Panel targets may also set ``title``, ``slot``, ``order``, ``icon``,
``route``, ``assets``, ``status``, ``api_prefix``, and ``router``. Those
fields are additive. ``assets`` is a local path inside the install, never
a URL. The host serves files from ``/plugins/<name>/`` and mounts ``router``
at ``/api/plugins/<name>/``.
"""

from __future__ import annotations

import inspect
import re
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PLUGIN_API_VERSION = "0.2.0"

PANEL_SLOTS = ("evaluate", "models", "knowledge", "finetune", "safety", "more")
_EVAL_TABS = ("connect", "run", "results", "compare", "judges")
_HOST_ONLY_ROUTES = {"#/evaluate", "#/jobs", "#/settings"} | {f"#/evaluate/{name}" for name in _EVAL_TABS}
_SLOT_HEADS = {"evaluate", "models", "knowledge", "finetune", "safety", "more", "jobs", "settings"}
_SAFE_NAME = re.compile(r"[A-Za-z0-9_.-]{1,64}")
_ICON_URL = re.compile(r"(https?:|//|javascript:|data:|expression\s*\()", re.IGNORECASE)
_ICON_HANDLER = re.compile(r"\son[a-z0-9_-]+\s*=", re.IGNORECASE)
_ICON_TAGS = ("script", "foreignobject", "iframe", "object", "embed", "link", "meta", "image", "audio", "video")

GROUPS = {
    "suites": "llm_eval.suites",
    "scorers": "llm_eval.scorers",
    "reporters": "llm_eval.reporters",
    "connectors": "llm_eval.connectors",
    "installers": "llm_eval.installers",
    "panels": "llm_eval.panels",
}

_RAN: ContextVar[list[dict] | None] = ContextVar("llm_eval_plugins_ran", default=None)


def parse_api_version(text: str | None) -> tuple[int, ...] | None:
    raw = str(text or "").strip()
    if raw.lower().startswith("v"):
        raw = raw[1:]
    if not raw:
        return None
    numbers = []
    for part in raw.split("."):
        if not part.isdigit():
            return None
        numbers.append(int(part))
    return tuple(numbers) if numbers else None


def api_compatible(declared: str | None, host: str = PLUGIN_API_VERSION) -> tuple[bool, str]:
    """True when both versions share a major component."""
    plugin_version = parse_api_version(declared)
    host_version = parse_api_version(host)
    if plugin_version is None:
        return False, f"unreadable plugin API version {declared!r}"
    if host_version is None:
        return False, f"unreadable host API version {host!r}"
    if plugin_version[0] != host_version[0]:
        return False, f"plugin API {declared} is not compatible with {host}"
    return True, ""


@dataclass(frozen=True)
class Plugin:
    group: str
    name: str
    version: str
    api_version: str
    distribution: str
    compatible: bool
    reason: str
    target: Any
    detail: dict | None = None

    def as_dict(self) -> dict:
        return {
            "group": self.group,
            "name": self.name,
            "version": self.version,
            "api_version": self.api_version,
            "distribution": self.distribution,
        }


def _text(obj: Any, key: str, default: str = "") -> str:
    if isinstance(obj, dict):
        value = obj.get(key, default)
    else:
        value = getattr(obj, key, default)
    return "" if value is None else str(value)


def _distribution_name(entry) -> str:
    dist = getattr(entry, "dist", None)
    if dist is None:
        return ""
    metadata = getattr(dist, "metadata", None)
    if metadata is not None:
        name = metadata.get("Name") if hasattr(metadata, "get") else None
        if name:
            return str(name)
    name = getattr(dist, "name", "")
    return str(name or "")


def _entry_points(group_id: str, supplied: Iterable | None):
    if supplied is not None:
        return list(supplied)
    from importlib.metadata import entry_points

    try:
        found = entry_points(group=group_id)
    except TypeError:
        found = entry_points().get(group_id, [])
    return list(found)


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _is_remote(value: str) -> bool:
    text = value.strip().lower()
    return text.startswith(("http://", "https://", "//"))


def _slot(target: Any) -> str:
    raw = _text(target, "slot", "more").strip().lower()
    return raw if raw in PANEL_SLOTS else "more"


def _order(target: Any) -> int:
    value = _get(target, "order", 100)
    if isinstance(value, bool):
        return 100
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return 100


def _route_token(name: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(name or "")).strip("-.")
    return token[:64] or "panel"


def _safe_plugin_name(name: str) -> bool:
    return bool(_SAFE_NAME.fullmatch(str(name or "")))


def sanitize_icon(raw: str | None) -> str:
    """Keep one inline SVG. Drop scripts, handlers, and external references."""
    text = str(raw or "").strip()
    if not text or len(text) > 8000:
        return ""
    lowered = text.lower()
    if lowered.count("<svg") != 1 or "</svg>" not in lowered:
        return ""
    if any(f"<{tag}" in lowered for tag in _ICON_TAGS):
        return ""
    scanned = re.sub(
        r"""\sxmlns(?::[a-z0-9]+)?\s*=\s*["']http://www\.w3\.org/[^"']+["']""",
        "",
        text,
        flags=re.IGNORECASE,
    )
    if _ICON_HANDLER.search(scanned) or _ICON_URL.search(scanned):
        return ""
    if not re.search(r"viewbox\s*=", text, re.IGNORECASE):
        return ""
    return text


def _module_dir(target: Any) -> Path | None:
    if isinstance(target, dict):
        return None
    try:
        subject = target if inspect.isclass(target) else target.__class__
        file = inspect.getfile(subject)
    except (TypeError, OSError):
        return None
    return Path(file).resolve().parent


def _resolve_path(value: Any, base: Path | None) -> Path | None:
    if isinstance(value, Path):
        path = value
    elif isinstance(value, str):
        if _is_remote(value):
            return None
        path = Path(value)
    else:
        return None
    if not path.is_absolute():
        if base is None:
            return None
        path = base / path
    return path


def _files_in_dir(path: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    js = path / "panel.js"
    css = path / "panel.css"
    if js.is_file():
        found["js"] = js.resolve()
    if css.is_file():
        found["css"] = css.resolve()
    return found


def _asset_files(target: Any) -> dict[str, Path]:
    raw = _get(target, "assets", None)
    if raw is None or raw == "":
        return {}
    if isinstance(raw, str) and _is_remote(raw):
        return {}
    base = _module_dir(target)
    if isinstance(raw, (str, Path)):
        path = _resolve_path(raw, base)
        if path is None or not path.is_dir():
            return {}
        return _files_in_dir(path)
    if not isinstance(raw, dict):
        return {}
    for key in ("js", "css", "root"):
        value = raw.get(key)
        if isinstance(value, str) and _is_remote(value):
            return {}
    found: dict[str, Path] = {}
    for key in ("js", "css"):
        path = _resolve_path(raw.get(key), base)
        if path is None or not path.is_file():
            continue
        if path.suffix.lower() not in {".js", ".css", ".mjs"}:
            continue
        found[key] = path.resolve()
    if found:
        return found
    root = _resolve_path(raw.get("root"), base)
    if root is not None and root.is_dir():
        return _files_in_dir(root)
    return {}


def _assets_rejected(target: Any) -> bool:
    raw = _get(target, "assets", None)
    if isinstance(raw, str):
        return _is_remote(raw)
    if isinstance(raw, dict):
        return any(isinstance(raw.get(key), str) and _is_remote(raw.get(key)) for key in ("js", "css", "root"))
    return False


def _clean_prefix(raw: Any) -> str:
    text = str(raw or "").strip()
    if not text or "://" in text or text.startswith("//") or not text.startswith("/"):
        return ""
    if any(ch in text for ch in " <>\"'?"):
        return ""
    return text.rstrip("/")


def _clean_route(raw: Any, slot: str, name: str) -> str:
    text = str(raw or "").strip()
    if text.startswith("#/") and "://" not in text and not text.startswith("#//"):
        if not any(ch in text for ch in " <>\"'"):
            return text
    token = _route_token(name)
    if slot == "more":
        return f"#/more/{token}"
    if slot == "evaluate":
        return f"#/evaluate/{token}"
    return f"#/{slot}"


def _status_payload(target: Any) -> dict | None:
    fn = _get(target, "status", None)
    if isinstance(fn, dict):
        raw = fn
    elif callable(fn):
        try:
            raw = fn()
        except Exception as exc:  # noqa: BLE001 - status must not take down the host
            return {
                "state": "missing",
                "checks": [
                    {
                        "id": "status",
                        "label": "Status",
                        "ok": False,
                        "detail": str(exc),
                        "fix": "",
                    }
                ],
            }
    else:
        return None
    if not isinstance(raw, dict):
        return {"state": "missing", "checks": []}
    state = str(raw.get("state") or "")
    if state not in {"ready", "partial", "missing"}:
        state = "missing"
    checks = []
    for row in raw.get("checks") or []:
        if not isinstance(row, dict):
            continue
        checks.append(
            {
                "id": str(row.get("id") or ""),
                "label": str(row.get("label") or ""),
                "ok": bool(row.get("ok")),
                "detail": str(row.get("detail") or ""),
                "fix": str(row.get("fix") or ""),
            }
        )
    return {"state": state, "checks": checks}


def _panel_detail(target: Any, *, call_status: bool) -> dict:
    name = _text(target, "name") or "panel"
    slot = _slot(target)
    return {
        "title": _text(target, "title", name) or name,
        "slot": slot,
        "order": _order(target),
        "icon": sanitize_icon(_get(target, "icon", "")),
        "route": _clean_route(_get(target, "route", ""), slot, name),
        "api_prefix": _clean_prefix(_get(target, "api_prefix", "")),
        "has_router": _get(target, "router", None) is not None,
        "status": _status_payload(target) if call_status else None,
        "assets_rejected": _assets_rejected(target),
        "_files": _asset_files(target),
    }


def _claim_ok(route: str, slot: str) -> bool:
    if not route.startswith("#/"):
        return False
    parts = [part for part in route[2:].split("/") if part]
    if not parts:
        return False
    head = parts[0]
    if route in _HOST_ONLY_ROUTES or head in {"jobs", "settings"}:
        return False
    if head in _SLOT_HEADS and head != slot:
        return False
    return True


def _fallback_route(slot: str, name: str, used: set[str]) -> str:
    token = _route_token(name)
    if slot in {"models", "knowledge", "finetune", "safety"}:
        home = f"#/{slot}"
        if home not in used:
            return home
    candidate = f"#/{slot}/{token}"
    if candidate not in used and _claim_ok(candidate, slot):
        return candidate
    candidate = f"#/more/{token}"
    suffix = 2
    while candidate in used:
        candidate = f"#/more/{token}-{suffix}"
        suffix += 1
    return candidate


def _assign_routes(rows: list[dict]) -> None:
    used = set(_HOST_ONLY_ROUTES)
    ordered = sorted(rows, key=lambda row: (int(row.get("order") or 100), str(row.get("name") or "")))
    for row in ordered:
        route = str(row.get("route") or "")
        slot = str(row.get("slot") or "more")
        if route in used or not _claim_ok(route, slot):
            route = _fallback_route(slot, str(row.get("name") or "panel"), used)
        row["route"] = route
        used.add(route)


def _public_assets(name: str, files: dict[str, Path]) -> dict | None:
    if not files or not _safe_plugin_name(name):
        return None
    return {key: f"/plugins/{name}/{path.name}" for key, path in files.items()}


def _public_row(row: dict) -> dict:
    return {key: value for key, value in row.items() if not str(key).startswith("_")}


def discover(group: str, entry_points: Iterable | None = None) -> list[Plugin]:
    """Load one group. Incompatible plugins are returned with ``compatible`` false."""
    if group not in GROUPS:
        known = ", ".join(GROUPS)
        raise KeyError(f"Unknown plugin group '{group}'. Known groups: {known}.")
    loaded = []
    for entry in _entry_points(GROUPS[group], entry_points):
        name = str(getattr(entry, "name", "") or "")
        distribution = _distribution_name(entry)
        try:
            target = entry.load()
        except Exception as exc:  # noqa: BLE001 - a bad plugin must not stop the host
            loaded.append(
                Plugin(group, name, "", "", distribution, False, f"failed to load: {exc}", None, None)
            )
            continue
        plugin_name = _text(target, "name", name) or name
        version = _text(target, "version")
        api_version = _text(target, "api_version")
        ok, reason = api_compatible(api_version)
        detail = _panel_detail(target, call_status=ok) if group == "panels" else None
        loaded.append(
            Plugin(
                group,
                plugin_name,
                version,
                api_version,
                distribution,
                ok,
                reason,
                target if ok else None,
                detail,
            )
        )
    return loaded


def usable(group: str, entry_points: Iterable | None = None) -> list[Plugin]:
    return [plugin for plugin in discover(group, entry_points) if plugin.compatible and plugin.target is not None]


def find(group: str, name: str, entry_points: Iterable | None = None) -> Plugin | None:
    for plugin in usable(group, entry_points):
        if plugin.name == name:
            return plugin
    return None


def compatible_names(group: str, entry_points: Iterable | None = None) -> list[str]:
    return [plugin.name for plugin in usable(group, entry_points)]


def reset_manifest() -> None:
    _RAN.set([])


def note_ran(plugin: Plugin) -> None:
    """Record one compatible plugin on the current run manifest."""
    if not plugin.compatible:
        return
    rows = _RAN.get()
    if rows is None:
        rows = []
        _RAN.set(rows)
    row = plugin.as_dict()
    key = (row["group"], row["name"], row["version"])
    if key in {(item["group"], item["name"], item["version"]) for item in rows}:
        return
    rows.append(row)


def current_manifest() -> list[dict]:
    return list(_RAN.get() or [])


def emit_reporters(output_dir: str, summary: dict | None = None, entry_points: Iterable | None = None) -> list[str]:
    """Call each compatible reporter and record it on the run manifest."""
    written = []
    for plugin in usable("reporters", entry_points):
        target = plugin.target
        writer = target.write if hasattr(target, "write") else target if callable(target) else None
        if writer is None:
            continue
        writer(output_dir, summary)
        note_ran(plugin)
        written.append(plugin.name)
    return written


def panel_catalog(entry_points: Iterable | None = None) -> list[dict]:
    """Every panel entry point, including incompatible and failed loads.

    Private ``_files`` maps asset keys to local paths for the host. API
    responses use :func:`panels`, which drops those paths.
    """
    rows = []
    for plugin in discover("panels", entry_points):
        detail = dict(plugin.detail or {})
        files = detail.pop("_files", {}) or {}
        if plugin.detail is None:
            token = _route_token(plugin.name or "panel")
            rows.append(
                {
                    "group": "panels",
                    "name": plugin.name or token,
                    "version": plugin.version,
                    "api_version": plugin.api_version,
                    "distribution": plugin.distribution,
                    "title": plugin.name or "Panel",
                    "slot": "more",
                    "order": 100,
                    "icon": "",
                    "route": f"#/more/{token}",
                    "assets": None,
                    "api_prefix": "",
                    "has_router": False,
                    "status": None,
                    "assets_rejected": False,
                    "compatible": False,
                    "reason": plugin.reason,
                    "availability": "error",
                    "_files": {},
                }
            )
            continue
        row = plugin.as_dict()
        row.update(detail)
        row["title"] = detail.get("title") or plugin.name
        row["compatible"] = bool(plugin.compatible)
        row["reason"] = plugin.reason
        if not plugin.compatible:
            row["status"] = None
            row["availability"] = "incompatible"
            row["assets"] = None
            row["_files"] = {}
        else:
            state = row["status"].get("state") if isinstance(row.get("status"), dict) else None
            row["availability"] = state if state in {"partial", "missing"} else "ready"
            row["assets"] = _public_assets(plugin.name, files)
            row["_files"] = files if _safe_plugin_name(plugin.name) else {}
        rows.append(row)
    _assign_routes(rows)
    rows.sort(key=lambda row: (int(row.get("order") or 100), str(row.get("name") or "")))
    return rows


def panels(entry_points: Iterable | None = None) -> list[dict]:
    """Compatible panels, with manifest fields and no filesystem paths."""
    return [_public_row(row) for row in panel_catalog(entry_points) if row.get("compatible")]


def installer_rows(entry_points: Iterable | None = None) -> list[dict]:
    """Installer profiles that this bundle build applied."""
    rows = []
    for plugin in usable("installers", entry_points):
        note_ran(plugin)
        row = plugin.as_dict()
        wheels = getattr(plugin.target, "wheels", None)
        row["wheels"] = list(wheels()) if callable(wheels) else []
        rows.append(row)
    return rows
