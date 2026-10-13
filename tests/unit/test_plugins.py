"""Plugin API 0.2.0. Entry points are fakes. No package index and no model server."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from llm_eval.models.base import BaseModel, ModelResponse
from llm_eval.plugins import (
    GROUPS,
    PLUGIN_API_VERSION,
    api_compatible,
    current_manifest,
    discover,
    emit_reporters,
    panels,
    reset_manifest,
    usable,
)
from llm_eval.runner import _get_evaluator_class, _get_model_class
from llm_eval_suite.app import create_app
from llm_eval_suite.runs import RunManager
from llm_eval_suite.suites import runner_for

_SPEC = importlib.util.spec_from_file_location(
    "sample_eval",
    Path(__file__).resolve().parents[1] / "plugins" / "sample_eval.py",
)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
_SPEC.loader.exec_module(_MODULE)
sample = _MODULE.sample
future = _MODULE.future


class _Dist:
    metadata = {"Name": "sample-plugin"}


class _Entry:
    def __init__(self, name, obj):
        self.name = name
        self.dist = _Dist()
        self._obj = obj

    def load(self):
        return self._obj


class _Connector:
    name = "sample_connector"
    version = "0.3.0"
    api_version = "0.2.0"

    def __init__(self, name, params):
        self.model_name = name
        self.params = params


class _Scorer:
    name = "sample_rubric"
    version = "0.2.0"
    api_version = "0.2.0"


def _entries(*pairs):
    return [_Entry(name, obj) for name, obj in pairs]


def test_major_version_gates_compatibility():
    assert PLUGIN_API_VERSION == "0.2.0"
    assert set(GROUPS) == {"suites", "scorers", "reporters", "connectors", "installers", "panels"}
    assert api_compatible("0.2.0")[0] is True
    assert api_compatible("0.1.0")[0] is True
    assert api_compatible("v0.9")[0] is True
    ok, reason = api_compatible("1.0.0")
    assert ok is False
    assert "1.0.0" in reason
    assert api_compatible("nope")[0] is False


def test_sample_plugin_loads_and_a_future_api_does_not():
    entries = _entries(("sample_probe", sample), ("future_probe", future))
    found = discover("suites", entries)
    by_name = {row.name: row for row in found}
    assert by_name["sample_probe"].compatible is True
    assert by_name["sample_probe"].version == "0.1.0"
    assert by_name["sample_probe"].distribution == "sample-plugin"
    assert by_name["future_probe"].compatible is False
    assert by_name["future_probe"].target is None
    assert [row.name for row in usable("suites", entries)] == ["sample_probe"]


def test_suite_runner_records_the_sample_on_the_manifest(monkeypatch: pytest.MonkeyPatch):
    plugin = discover("suites", _entries(("sample_probe", sample)))[0]

    def fake_find(group, name, entry_points=None):
        if group == "suites" and name == "sample_probe":
            return plugin
        return None

    monkeypatch.setattr("llm_eval.plugins.find", fake_find)
    reset_manifest()
    runner = runner_for("sample_probe")
    assert runner.run(None, {})["source"] == "plugin"
    manifest = current_manifest()
    assert manifest == [
        {
            "group": "suites",
            "name": "sample_probe",
            "version": "0.1.0",
            "api_version": "0.2.0",
            "distribution": "sample-plugin",
        }
    ]
    with pytest.raises(KeyError):
        runner_for("missing_suite")


def test_connector_scorer_reporter_and_panel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    connector = discover("connectors", _entries(("sample_connector", _Connector)))[0]
    scorer = discover("scorers", _entries(("sample_rubric", _Scorer)))[0]

    def fake_find(group, name, entry_points=None):
        if group == "connectors" and name == "sample_connector":
            return connector
        if group == "scorers" and name == "sample_rubric":
            return scorer
        return None

    monkeypatch.setattr("llm_eval.plugins.find", fake_find)
    reset_manifest()
    assert _get_model_class("sample_connector") is _Connector
    assert _get_evaluator_class("sample_rubric") is _Scorer
    names = {row["name"] for row in current_manifest()}
    assert names == {"sample_connector", "sample_rubric"}

    written = emit_reporters(tmp_path, {"ok": True}, _entries(("sample_probe", sample)))
    assert written == ["sample_probe"]
    assert (tmp_path / "sample-plugin.txt").read_text(encoding="utf-8") == "sample\n"
    assert panels(_entries(("sample_probe", sample)))[0]["title"] == "Sample panel"


def test_click_through_run_stores_the_plugin_manifest(tmp_path: Path):
    dataset = tmp_path / "facts.jsonl"
    dataset.write_text(
        json.dumps({"prompt": "Capital of France?", "expected_answer": "Paris"}) + "\n",
        encoding="utf-8",
    )

    class Stub(BaseModel):
        def __init__(self):
            super().__init__("stub", {})

        def generate(self, prompt: str, **kwargs) -> ModelResponse:
            return ModelResponse(text="Paris", latency_ms=1.0, tokens_used=1, metadata={})

    manager = RunManager(tmp_path / "runs", model_factory=lambda _profile: Stub())
    record = manager.start(
        connection={"type": "ollama", "model": "stub", "base_url": "http://127.0.0.1:11434"},
        preset_id="quant_quick",
        dataset_path=str(dataset),
        background=False,
    )
    assert record["plugins"] == []


def test_extra_wheels_and_installer_profile_are_in_the_bundle(tmp_path: Path):
    from llm_eval.bundle_offline import BUNDLE_DIRNAME, build_offline_bundle
    from llm_eval.offline import DETECTOR_PINS
    from standard_bench.bundle import build_bundle

    root = Path(__file__).resolve().parents[2]
    kit = tmp_path / "kit.zip"
    build_bundle(kit, root / "standard_bench" / "fixtures", repo_root=root)
    wheels = tmp_path / "wheels"
    wheels.mkdir()
    (wheels / "demo-1.0-py3-none-any.whl").write_bytes(b"pure")
    private = tmp_path / "acme_eval-1.2.0-py3-none-any.whl"
    private.write_bytes(b"private-wheel")
    detectors = tmp_path / "detectors"
    for pin in DETECTOR_PINS:
        folder = detectors / pin.key
        folder.mkdir(parents=True)
        for name in pin.files:
            (folder / name).write_text("x\n", encoding="utf-8")
    output = tmp_path / "bundle.tar.gz"
    build_offline_bundle(
        output,
        repo_root=root,
        wheelhouse=wheels,
        download=False,
        kit=kit,
        fetch=False,
        detectors=detectors,
        extra_wheels=[private],
        installer_entry_points=_entries(("sample_probe", sample)),
    )
    import tarfile

    with tarfile.open(output, "r:gz") as archive:
        member = archive.extractfile(BUNDLE_DIRNAME + "/wheelhouse/common/acme_eval-1.2.0-py3-none-any.whl")
        assert member is not None
        assert member.read() == b"private-wheel"
        manifest = json.loads(archive.extractfile(BUNDLE_DIRNAME + "/manifest.json").read().decode("utf-8"))
    assert manifest["extra_wheels"] == ["common/acme_eval-1.2.0-py3-none-any.whl"]
    assert manifest["installers"][0]["name"] == "sample_probe"
    assert manifest["installers"][0]["version"] == "0.1.0"
    assert manifest["installers"][0]["api_version"] == "0.2.0"


def test_plugins_route_reports_the_api_version(tmp_path: Path):
    app = create_app(data_dir=tmp_path / "data", runs_dir=tmp_path / "runs")
    body = TestClient(app).get("/api/plugins").json()
    assert body["api_version"] == "0.2.0"
    assert body["groups"]["suites"] == "llm_eval.suites"
    assert body["panels"] == []
