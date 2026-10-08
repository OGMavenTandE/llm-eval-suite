import json
import os
from pathlib import Path

import pytest


class OutboundGuard:
    """Socket harness. A non-loopback connect fails immediately and is recorded."""

    def __init__(self) -> None:
        self.attempts: list = []

    def install(self, monkeypatch) -> None:
        import socket

        real_connect = socket.socket.connect
        real_create = socket.create_connection
        guard = self

        def connect(sock, address):
            guard.check(address)
            return real_connect(sock, address)

        def create_connection(address, *args, **kwargs):
            guard.check(address)
            return real_create(address, *args, **kwargs)

        monkeypatch.setattr(socket.socket, "connect", connect)
        monkeypatch.setattr(socket, "create_connection", create_connection)

    def check(self, address) -> None:
        if not isinstance(address, tuple) or not address:
            return
        host = str(address[0])
        if host in {"127.0.0.1", "::1", "localhost"} or host.startswith("127."):
            return
        self.attempts.append(address)
        raise OSError(f"outbound connection blocked: {address}")


@pytest.fixture
def outbound_guard(monkeypatch):
    guard = OutboundGuard()
    guard.install(monkeypatch)
    return guard


@pytest.fixture(autouse=True)
def _reset_offline_state():
    """Keep one test's offline switch from leaking into the next test."""
    from llm_eval import offline

    offline.reset_for_tests()
    saved = {key: os.environ.get(key) for key in offline.OFFLINE_ENV}
    yield
    offline.reset_for_tests()
    for key, previous in saved.items():
        if previous is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = previous


@pytest.fixture
def sample_dataset_path(tmp_path: Path) -> Path:
    rows = [
        {"prompt": "What is 2+2?", "expected_answer": "4"},
        {"prompt": "Capital of France?", "expected_answer": "Paris"},
    ]
    path = tmp_path / "sample.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


@pytest.fixture
def invalid_dataset_path(tmp_path: Path) -> Path:
    rows = [{"prompt": "Missing expected answer field"}]
    path = tmp_path / "invalid.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    return path


@pytest.fixture
def valid_config(sample_dataset_path: Path) -> dict:
    return {
        "run_name": "test-run",
        "output_dir": str(sample_dataset_path.parent / "results"),
        "dataset": str(sample_dataset_path),
        "models": [
            {
                "name": "test-model",
                "provider": "ollama",
                "params": {"temperature": 0.0},
            }
        ],
        "evaluators": [
            {"name": "correctness", "mode": "exact_match", "threshold": 0.8},
            {"name": "latency", "max_ms": 5000},
        ],
    }
