"""Offline bundle and air-gap doctor. No package index, dataset fetch, or GPU."""

from __future__ import annotations

import io
import json
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from llm_eval.bundle_offline import (
    BUNDLE_DIRNAME,
    BundleOfflineError,
    JUDGE_TRADEOFF,
    build_offline_bundle,
    constrained_commands,
    host_platform_tag,
    linux_download_plan,
    render_install_sh,
)
from llm_eval.doctor import PROXY_ENV, Check, DoctorError, format_report, judge_present, open_bundle, probe_egress, run_doctor
from llm_eval.offline import DETECTOR_PINS
from llm_eval_suite.eval_env import CUDA_INDEX, GARAK_VERSION, TORCH_VERSION, TRANSFORMERS_VERSION

ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "standard_bench" / "fixtures"


def _kit(tmp_path: Path) -> Path:
    from standard_bench.bundle import build_bundle

    destination = tmp_path / "standard-capability-kit.zip"
    build_bundle(destination, FIXTURES, repo_root=ROOT)
    return destination


def _detectors(tmp_path: Path) -> Path:
    root = tmp_path / "detectors"
    for pin in DETECTOR_PINS:
        folder = root / pin.key
        folder.mkdir(parents=True)
        for name in pin.files:
            (folder / name).write_text("placeholder\n", encoding="utf-8")
    return root


def _wheels(tmp_path: Path) -> Path:
    root = tmp_path / "wheels"
    root.mkdir()
    (root / "torch-2.13.0-cp311-cp311-manylinux_2_28_x86_64.whl").write_bytes(b"torch-wheel")
    (root / "demo-1.0-py3-none-any.whl").write_bytes(b"pure-wheel")
    return root


def _judges(tmp_path: Path) -> Path:
    root = tmp_path / "judge-models"
    for name in ("qwen2.5-3b-instruct", "llama3.2-3b"):
        folder = root / name
        folder.mkdir(parents=True)
        (folder / "config.json").write_text("{}\n", encoding="utf-8")
    return root


def _bundle(tmp_path: Path, *, judges: bool = False) -> Path:
    output = tmp_path / "llm-eval-offline.tar.gz"
    build_offline_bundle(
        output,
        repo_root=ROOT,
        wheelhouse=_wheels(tmp_path),
        download=False,
        kit=_kit(tmp_path),
        fetch=False,
        detectors=_detectors(tmp_path),
        judges_dir=_judges(tmp_path) if judges else None,
    )
    return output


def _members(path: Path) -> dict[str, bytes]:
    with tarfile.open(path, "r:gz") as archive:
        found = {}
        for info in archive.getmembers():
            if not info.isfile():
                continue
            handle = archive.extractfile(info)
            found[info.name] = b"" if handle is None else handle.read()
        return found


def _clear_network_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in PROXY_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("LLM_EVAL_JUDGES_DIR", raising=False)
    empty = tmp_path / "no-ollama"
    empty.mkdir(exist_ok=True)
    monkeypatch.setenv("OLLAMA_MODELS", str(empty))


def _blocked() -> tuple[bool, str]:
    return True, "no route to 1.1.1.1:443 (TimeoutError)"


def _open() -> tuple[bool, str]:
    return False, "connected to 1.1.1.1:443"


def test_windows_commands_pin_the_platform():
    commands = constrained_commands(Path("win"), "win-amd64", "3.11")
    joined = "\n".join(" ".join(command) for command in commands)
    assert "win_amd64" in joined
    assert "--only-binary=:all:" in joined
    assert "cp311" in joined
    assert CUDA_INDEX in joined
    assert f"torch=={TORCH_VERSION}" in joined
    assert f"garak=={GARAK_VERSION}" in joined
    assert f"transformers=={TRANSFORMERS_VERSION}" in joined
    assert ".[api]" not in joined


def test_linux_plan_follows_the_builder():
    version = f"{sys.version_info.major}.{sys.version_info.minor}"
    mode, commands = linux_download_plan(Path("wheels"), version)
    joined = "\n".join(" ".join(command) for command in commands)
    assert TORCH_VERSION in joined
    assert CUDA_INDEX in joined
    if host_platform_tag() == "linux-x86_64":
        assert mode == "host"
        assert ".[api]" in joined
    else:
        assert mode == "constrained"
        assert "manylinux2014_x86_64" in joined


def test_unknown_python_is_rejected():
    with pytest.raises(BundleOfflineError, match="Unsupported"):
        constrained_commands(Path("wheels"), "linux-x86_64", "3.9")


def test_bundle_packs_wheels_kit_detectors_and_a_judge_manifest(tmp_path: Path):
    archive_path = _bundle(tmp_path)
    members = _members(archive_path)
    prefix = BUNDLE_DIRNAME + "/"
    names = set(members)
    assert prefix + "install.sh" in names
    assert prefix + "SHA256SUMS" in names
    assert prefix + "manifest.json" in names
    assert prefix + "kit/standard-capability-kit.zip" in names
    assert prefix + "wheelhouse/linux-x86_64/torch-2.13.0-cp311-cp311-manylinux_2_28_x86_64.whl" in names
    assert prefix + "wheelhouse/common/demo-1.0-py3-none-any.whl" in names
    assert prefix + "garak_detectors/DETECTORS.json" in names
    assert prefix + "garak_detectors/roberta_nli/config.json" in names
    assert prefix + "judges/JUDGES.json" in names
    judges = json.loads(members[prefix + "judges/JUDGES.json"].decode("utf-8"))
    assert judges["included"] is False
    assert judges["bytes"] == 0
    assert "qwen2.5:3b-instruct" in judges["tradeoff"]
    assert "about 4 GB" in judges["tradeoff"]
    assert JUDGE_TRADEOFF in judges["tradeoff"]
    detectors = json.loads(members[prefix + "garak_detectors/DETECTORS.json"].decode("utf-8"))
    assert all(row["included"] for row in detectors["detectors"])
    manifest = json.loads(members[prefix + "manifest.json"].decode("utf-8"))
    assert manifest["pins"]["torch"] == TORCH_VERSION
    assert manifest["platforms"]["linux-x86_64"]["present"] is True
    assert manifest["platforms"]["win-amd64"]["present"] is False
    assert manifest["judges"]["included"] is False
    sums = members[prefix + "SHA256SUMS"].decode("utf-8")
    assert "manifest.json" in sums
    script = members[prefix + "install.sh"].decode("utf-8")
    assert script.startswith("#!/bin/sh\n")
    assert "--no-index" in script
    assert "PIP_NO_INDEX=1" in script
    assert "python -m llm_eval doctor --bundle" in script
    lowered = script.lower()
    assert "curl" not in lowered
    assert "wget" not in lowered
    assert "https://" not in lowered
    assert "http://" not in lowered
    with tarfile.open(archive_path, "r:gz") as archive:
        info = archive.getmember(prefix + "install.sh")
        assert info.mode & 0o111


def test_bundle_can_embed_judge_files(tmp_path: Path):
    archive_path = _bundle(tmp_path, judges=True)
    members = _members(archive_path)
    judges = json.loads(members[BUNDLE_DIRNAME + "/judges/JUDGES.json"].decode("utf-8"))
    assert judges["included"] is True
    assert judges["bytes"] > 0
    assert BUNDLE_DIRNAME + "/judges/qwen2.5-3b-instruct/config.json" in members


def test_missing_wheelhouse_is_an_error(tmp_path: Path):
    with pytest.raises(BundleOfflineError, match="wheelhouse"):
        build_offline_bundle(
            tmp_path / "out.tar.gz",
            repo_root=ROOT,
            download=False,
            kit=_kit(tmp_path),
            fetch=False,
            detectors=_detectors(tmp_path),
        )


def test_windows_download_failure_still_packs_linux(tmp_path: Path):
    def runner(command, cwd, check=False, capture_output=True, text=True):
        if "win_amd64" in command:
            return subprocess.CompletedProcess(command, 1, "", "no windows wheel for this package")
        if "-d" in command:
            dest = Path(command[command.index("-d") + 1])
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "sample-1.0-py3-none-any.whl").write_bytes(b"wheel")
        return subprocess.CompletedProcess(command, 0, "", "")

    output = tmp_path / "out.tar.gz"
    summary = build_offline_bundle(
        output,
        repo_root=ROOT,
        download=True,
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}",
        kit=_kit(tmp_path),
        fetch=False,
        detectors=_detectors(tmp_path),
        runner=runner,
    )
    assert summary["platforms"]["linux-x86_64"]["present"] is True
    assert summary["platforms"]["win-amd64"]["present"] is False
    assert "no windows wheel" in summary["platforms"]["win-amd64"]["reason"]
    assert output.is_file()


def test_install_sh_mentions_the_pinned_versions():
    script = render_install_sh()
    assert f"torch=={TORCH_VERSION}" in script
    assert f"garak=={GARAK_VERSION}" in script
    assert f"transformers=={TRANSFORMERS_VERSION}" in script


def test_probe_egress_treats_a_blocked_socket_as_closed(monkeypatch: pytest.MonkeyPatch):
    class Blocked:
        def settimeout(self, _timeout):
            return None

        def connect(self, _address):
            raise TimeoutError("timed out")

        def close(self):
            return None

    monkeypatch.setattr("llm_eval.doctor.socket.socket", lambda *_args, **_kwargs: Blocked())
    blocked, detail = probe_egress()
    assert blocked is True
    assert "1.1.1.1:443" in detail


def test_probe_egress_reports_an_open_socket(monkeypatch: pytest.MonkeyPatch):
    class Open:
        def settimeout(self, _timeout):
            return None

        def connect(self, _address):
            return None

        def close(self):
            return None

    monkeypatch.setattr("llm_eval.doctor.socket.socket", lambda *_args, **_kwargs: Open())
    blocked, detail = probe_egress()
    assert blocked is False
    assert "connected" in detail


def test_doctor_refuses_a_path_that_escapes_the_bundle(tmp_path: Path):
    archive_path = tmp_path / "bad.tar.gz"
    payload = b"nope"
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo("../evil.txt")
        info.size = len(payload)
        archive.addfile(info, io.BytesIO(payload))
    with pytest.raises(DoctorError, match="Refusing"):
        open_bundle(archive_path)


def test_judge_present_reads_an_ollama_manifest(tmp_path: Path):
    manifest = tmp_path / "manifests" / "registry.ollama.ai" / "library" / "qwen2.5" / "3b-instruct"
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}\n", encoding="utf-8")
    assert judge_present(tmp_path, "qwen2.5:3b-instruct") is True
    assert judge_present(tmp_path, "llama3.2:3b") is False


def test_doctor_passes_a_closed_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _clear_network_env(monkeypatch, tmp_path)
    archive_path = _bundle(tmp_path, judges=True)
    checks = run_doctor(
        archive_path,
        egress_probe=_blocked,
        gpu_lookup=lambda: None,
    )
    report = format_report(checks)
    assert "OVERALL PASS" in report
    assert "FAIL" not in report
    by_name = {check.name: check for check in checks}
    assert by_name["hashes"].ok
    assert by_name["kit"].ok
    assert by_name["detectors"].ok
    assert by_name["judges"].ok
    assert by_name["python"].ok
    assert "does not require" in by_name["gpu"].detail


def test_doctor_fails_open_egress_proxy_hashes_and_a_required_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    _clear_network_env(monkeypatch, tmp_path)
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example:8080")
    archive_path = _bundle(tmp_path, judges=False)
    unpacked = tmp_path / "unpacked"
    with tarfile.open(archive_path, "r:gz") as archive:
        extract_kwargs = {"filter": "data"} if sys.version_info >= (3, 12) else {}
        archive.extractall(unpacked, **extract_kwargs)
    root = unpacked / BUNDLE_DIRNAME
    wheel = root / "wheelhouse" / "common" / "demo-1.0-py3-none-any.whl"
    wheel.write_bytes(b"tampered")
    checks = run_doctor(
        root,
        egress_probe=_open,
        gpu_lookup=lambda: None,
        require_gpu=True,
    )
    by_name = {check.name: check for check in checks}
    assert by_name["egress"].ok is False
    assert "not air-gapped" in by_name["egress"].detail
    assert by_name["proxy"].ok is False
    assert "HTTPS_PROXY" in by_name["proxy"].detail
    assert by_name["hashes"].ok is False
    assert by_name["judges"].ok is False
    assert "qwen2.5:3b-instruct" in by_name["judges"].detail
    assert by_name["gpu"].ok is False
    report = format_report(checks)
    assert report.endswith("OVERALL FAIL\n")
    assert isinstance(checks[0], Check)


def test_cli_bundle_and_doctor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    _clear_network_env(monkeypatch, tmp_path)
    monkeypatch.setattr("llm_eval.doctor.probe_egress", lambda: _blocked())
    monkeypatch.setattr("llm_eval.doctor.default_gpu_name", lambda: "Test GPU")
    output = tmp_path / "bundle.tar.gz"
    from llm_eval.cli import main

    with pytest.raises(SystemExit) as built:
        main(
            [
                "bundle-offline",
                "--output",
                str(output),
                "--no-download",
                "--no-fetch",
                "--wheelhouse",
                str(_wheels(tmp_path)),
                "--kit",
                str(_kit(tmp_path)),
                "--detectors",
                str(_detectors(tmp_path)),
                "--judges-dir",
                str(_judges(tmp_path)),
                "--repo-root",
                str(ROOT),
            ]
        )
    assert built.value.code == 0
    text = capsys.readouterr().out
    assert "judges included" in text
    assert str(output) in text

    with pytest.raises(SystemExit) as checked:
        main(["doctor", "--bundle", str(output), "--require-gpu"])
    assert checked.value.code == 0
    report = capsys.readouterr().out
    assert "OVERALL PASS" in report
    assert "Test GPU" in report

    with pytest.raises(SystemExit) as missing:
        main(["--config", str(tmp_path / "missing.yaml")])
    assert missing.value.code == 1


def test_airgap_doc_describes_the_bundle_and_keeps_the_manual_commands():
    from llm_eval_suite.eval_env import wheelhouse_commands

    text = Path("docs/AIRGAP_INSTALL.md").read_text(encoding="utf-8")
    assert "python -m llm_eval bundle-offline" in text
    assert "python -m llm_eval doctor" in text
    assert "./install.sh" in text
    assert "about 4 GB" in text
    assert "about 1.4 GB" in text
    assert "--require-gpu" in text
    for command in wheelhouse_commands():
        assert command in text
    assert "DoD" not in text
