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
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any, Iterable

PLUGIN_API_VERSION = "0.2.0"

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
                Plugin(group, name, "", "", distribution, False, f"failed to load: {exc}", None)
            )
            continue
        plugin_name = _text(target, "name", name) or name
        version = _text(target, "version")
        api_version = _text(target, "api_version")
        ok, reason = api_compatible(api_version)
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


def panels(entry_points: Iterable | None = None) -> list[dict]:
    rows = []
    for plugin in usable("panels", entry_points):
        row = plugin.as_dict()
        row["title"] = _text(plugin.target, "title", plugin.name) or plugin.name
        rows.append(row)
    return rows


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
