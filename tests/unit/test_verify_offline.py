"""Offline install check. Wheel packing stays on this machine. No dataset fetch."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from llm_eval.verify_offline import (
    EGRESS_ATTEMPT,
    _venv_python,
    closure,
    install_network_guard,
    network_violations,
    pack_distribution,
)

ROOT = Path(__file__).resolve().parents[2]


def test_network_violations_allow_only_the_egress_probe():
    text = "\n".join(
        [
            EGRESS_ATTEMPT,
            "('huggingface.co', 443)",
            "('127.0.0.1', 11434)",
        ]
    )
    assert network_violations(text, allow_egress=False) == [
        EGRESS_ATTEMPT,
        "('huggingface.co', 443)",
        "('127.0.0.1', 11434)",
    ]
    assert network_violations(text, allow_egress=True) == [
        "('huggingface.co', 443)",
        "('127.0.0.1', 11434)",
    ]
    assert network_violations("", allow_egress=True) == []


def test_edge_demo_quotes_the_missing_directory_and_the_noise():
    text = (ROOT / "docs" / "EDGE_DEMO.md").read_text(encoding="utf-8")
    assert "Standard benchmark directory does not exist:" in text
    assert "HTTP 429" in text
    assert "Retry-After" in text
    assert "51 against 50" in text
    assert "44 against 55" in text
    assert "MATH-500 went from 6 to 3" in text
    assert "within the 95% interval" in text
    assert "larger item count" in text
    fetch = (ROOT / "standard_bench" / "fetch.py").read_text(encoding="utf-8")
    assert "429" in fetch
    assert "_RETRY_STATUS = frozenset({429, 500, 502, 503, 504})" in fetch


def test_requests_closure_packs_and_installs_offline(tmp_path: Path):
    wheels = tmp_path / "wheels"
    names = closure(["requests"])
    assert "requests" in {name.lower() for name in names}
    for name in names:
        pack_distribution(name, wheels)
    assert list(wheels.glob("*.whl"))
    venv = tmp_path / "venv"
    created = subprocess.run([sys.executable, "-m", "venv", str(venv)], capture_output=True, text=True, check=False)
    assert created.returncode == 0, created.stderr
    python = _venv_python(venv)
    env = os.environ.copy()
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    installed = subprocess.run(
        [str(python), "-m", "pip", "install", "--no-index", "--find-links", str(wheels), "requests"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert installed.returncode == 0, installed.stderr[-1500:]
    probe = subprocess.run(
        [str(python), "-c", "import requests; print(requests.__version__)"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert probe.returncode == 0, probe.stderr
    site = subprocess.check_output(
        [str(python), "-c", "import site; print(site.getsitepackages()[0])"],
        text=True,
    ).strip()
    install_network_guard(Path(site))
    log = tmp_path / "network.log"
    blocked_env = os.environ.copy()
    blocked_env["LLM_EVAL_NETBLOCK_LOG"] = str(log)
    blocked = subprocess.run(
        [str(python), "-c", "import socket; socket.create_connection(('1.1.1.1', 443), timeout=2)"],
        capture_output=True,
        text=True,
        check=False,
        env=blocked_env,
    )
    assert blocked.returncode != 0
    assert "Offline verification blocked a network attempt" in (blocked.stderr or "")
    assert "Nothing was sent." in (blocked.stderr or "")
    assert EGRESS_ATTEMPT in log.read_text(encoding="utf-8")
