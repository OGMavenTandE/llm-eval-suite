"""Panel discovery, manifest fields, and host routing. No package index."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from llm_eval.plugins import panel_catalog, panels, sanitize_icon
from llm_eval_suite.app import create_app

_SAMPLE_PATH = Path(__file__).resolve().parents[1] / "plugins" / "sample_panel.py"
_SPEC = importlib.util.spec_from_file_location("sample_panel_mod", _SAMPLE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC is not None and _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
sample_panel = _MODULE.sample_panel


class _Dist:
    metadata = {"Name": "sample-panel"}


class _Entry:
    def __init__(self, name, obj):
        self.name = name
        self.dist = _Dist()
        self._obj = obj

    def load(self):
        return self._obj


class _Boom:
    name = "boom"
    dist = _Dist()

    def load(self):
        raise RuntimeError("missing wheel")


class _Old:
    name = "old_panel"
    version = "9.0.0"
    api_version = "1.2.0"
    title = "Old fine-tune"
    slot = "finetune"
    route = "#/finetune"
    order = 1


class _Partial:
    name = "partial_models"
    version = "0.1.0"
    api_version = "0.2.0"
    title = "Partial models"
    slot = "models"
    order = 1
    route = "#/models"

    def status(self):
        return {
            "state": "partial",
            "checks": [
                {
                    "id": "ollama",
                    "label": "Ollama",
                    "ok": False,
                    "detail": "Ollama isn't running.",
                    "fix": "ollama serve",
                }
            ],
        }


class _BadIcon:
    name = "bad_icon"
    version = "0.1.0"
    api_version = "0.2.0"
    title = "Bad icon"
    slot = "more"
    order = 40
    icon = '<svg viewBox="0 0 20 20"><script>alert(1)</script></svg>'


class _Remote:
    name = "remote_assets"
    version = "0.1.0"
    api_version = "0.2.0"
    title = "Remote"
    slot = "more"
    order = 50
    assets = "https://example.com/panel.js"


class _Early:
    name = "early_panel"
    version = "0.1.0"
    api_version = "0.2.0"
    title = "Early"
    slot = "more"
    order = 5


class _Late:
    name = "late_panel"
    version = "0.1.0"
    api_version = "0.2.0"
    title = "Late"
    slot = "more"
    order = 30


class _Thief:
    name = "thief"
    version = "0.1.0"
    api_version = "0.2.0"
    title = "Thief"
    slot = "more"
    order = 1
    route = "#/jobs"


def _entries(*pairs):
    return [_Entry(name, obj) for name, obj in pairs]


def test_icon_sanitizer_drops_scripts_and_remote_refs():
    assert sanitize_icon(sample_panel.icon).startswith("<svg")
    assert sanitize_icon('<svg viewBox="0 0 20 20"><script>x</script></svg>') == ""
    assert sanitize_icon('<svg viewBox="0 0 20 20" onclick="alert(1)"></svg>') == ""
    assert sanitize_icon('<svg viewBox="0 0 20 20" href="https://example.com/a.svg"></svg>') == ""
    assert sanitize_icon("<svg></svg>") == ""
    assert sanitize_icon("nope") == ""


def test_catalog_fields_order_and_rejected_routes():
    rows = panel_catalog(
        [
            *_entries(
                ("early", _Early()),
                ("late", _Late()),
                ("thief", _Thief()),
                ("bad", _BadIcon()),
                ("remote", _Remote()),
                ("old", _Old()),
                ("partial", _Partial()),
            ),
            _Boom(),
        ]
    )
    by_name = {row["name"]: row for row in rows}
    assert [row["name"] for row in rows if row["slot"] == "more" and row["compatible"]][:2] == [
        "thief",
        "early_panel",
    ]
    assert by_name["early_panel"]["order"] == 5
    assert by_name["late_panel"]["order"] == 30
    assert rows.index(by_name["early_panel"]) < rows.index(by_name["late_panel"])
    assert by_name["thief"]["route"] != "#/jobs"
    assert by_name["thief"]["route"].startswith("#/more/")
    assert by_name["bad_icon"]["icon"] == ""
    assert by_name["remote_assets"]["assets"] is None
    assert by_name["remote_assets"]["assets_rejected"] is True
    assert by_name["old_panel"]["compatible"] is False
    assert by_name["old_panel"]["availability"] == "incompatible"
    assert by_name["old_panel"]["slot"] == "finetune"
    assert "1.2.0" in by_name["old_panel"]["reason"]
    assert by_name["old_panel"]["status"] is None
    assert "target" not in by_name["old_panel"]
    assert by_name["boom"]["availability"] == "error"
    assert "missing wheel" in by_name["boom"]["reason"]
    assert by_name["partial_models"]["availability"] == "partial"
    assert by_name["partial_models"]["status"]["checks"][0]["fix"] == "ollama serve"
    public = panels(_entries(("partial", _Partial())))
    assert "_files" not in public[0]
    assert public[0]["title"] == "Partial models"


def test_empty_shell_reserves_toolkit_slots(tmp_path: Path):
    app = create_app(data_dir=tmp_path / "data", runs_dir=tmp_path / "runs", panel_entries=[])
    body = TestClient(app).get("/api/plugins").json()
    assert body["api_version"] == "0.2.0"
    assert body["panels"] == []
    nav = {slot["id"]: slot for slot in body["nav"]}
    assert list(nav) == ["evaluate", "models", "knowledge", "finetune", "safety", "jobs", "settings"]
    for slot_id in ("models", "knowledge", "finetune", "safety"):
        assert nav[slot_id]["availability"] == "not_installed"
        assert nav[slot_id]["tag"] == "Not installed"
        assert nav[slot_id]["command"].startswith("pip install --no-index --find-links ")
        assert "It runs on this computer only." == nav[slot_id]["local_note"]
    assert nav["finetune"]["command"].endswith("airgap-ai-toolkit[finetune]")
    assert nav["models"]["command"].endswith("airgap-ai-toolkit")
    assert "[rag]" in nav["knowledge"]["command"]
    assert "[safety]" in nav["safety"]["command"]
    assert "Restart this app" in body["restart_note"]
    assert body["environment"]
    page = TestClient(app).get("/")
    assert page.status_code == 200
    assert "Content-Security-Policy" in page.headers
    assert "default-src 'self'" in page.headers["Content-Security-Policy"]
    text = page.text.lower()
    assert "cdn" not in text
    assert "fonts.googleapis" not in text
    assert "/static/shell.js" in page.text
    assert "Not installed" in page.text
    assert 'data-slot="finetune"' in page.text
    assert ">1" in page.text and ">2" in page.text and ">3" in page.text


def test_sample_panel_route_assets_and_not_installed_override(tmp_path: Path):
    entries = _entries(("sample_panel", sample_panel), ("old", _Old()), ("partial", _Partial()))
    app = create_app(data_dir=tmp_path / "data", runs_dir=tmp_path / "runs", panel_entries=entries)
    client = TestClient(app)
    body = client.get("/api/plugins").json()
    catalog = {row["name"]: row for row in body["catalog"]}
    sample = catalog["sample_panel"]
    assert sample["slot"] == "more"
    assert sample["route"] == "#/sample"
    assert sample["availability"] == "ready"
    assert sample["assets"]["js"].endswith("/plugins/sample_panel/panel.js")
    assert sample["assets"]["css"].endswith("/plugins/sample_panel/panel.css")
    assert sample["mounted_at"] == "/api/plugins/sample_panel"
    assert sample["icon"].startswith("<svg")
    assert "<script" not in sample["icon"].lower()
    nav = {slot["id"]: slot for slot in body["nav"]}
    assert nav["more"]["route"] == "#/sample"
    assert nav["finetune"]["availability"] == "incompatible"
    assert nav["finetune"]["tag"] == "Update needed"
    assert nav["models"]["availability"] == "partial"
    assert nav["models"]["tag"] == ""
    assert nav["knowledge"]["availability"] == "not_installed"
    script = client.get("/plugins/sample_panel/panel.js")
    assert script.status_code == 200
    assert "export function mount" in script.text
    assert client.get("/plugins/sample_panel/panel.css").status_code == 200
    assert client.get("/plugins/sample_panel/missing.js").status_code == 404
    assert client.get("/plugins/sample_panel/../sample_panel.py").status_code == 404
    ping = client.get("/api/plugins/sample_panel/ping")
    assert ping.status_code == 200
    assert ping.json()["ok"] is True


def test_settings_toolkit_folder_changes_the_install_command(tmp_path: Path):
    app = create_app(data_dir=tmp_path / "data", runs_dir=tmp_path / "runs", panel_entries=[])
    client = TestClient(app)
    saved = client.put(
        "/api/settings",
        json={"toolkit_root": "/opt/lab/toolkit", "classification": {"enabled": True, "text": "UNCLASSIFIED", "color": "green"}},
    )
    assert saved.status_code == 200
    assert saved.json()["toolkit_root"] == "/opt/lab/toolkit"
    body = client.get("/api/plugins").json()
    finetune = next(slot for slot in body["nav"] if slot["id"] == "finetune")
    assert "/opt/lab/toolkit/wheels" in finetune["command"]
    assert body["classification"]["text"] == "UNCLASSIFIED"
    assert body["classification"]["enabled"] is True
    refused = client.put("/api/settings", json={"toolkit_root": "https://example.com/wheels"})
    assert refused.status_code == 400
    bad_color = client.put(
        "/api/settings",
        json={"classification": {"enabled": True, "text": "SECRET", "color": "pink"}},
    )
    assert bad_color.status_code == 400
