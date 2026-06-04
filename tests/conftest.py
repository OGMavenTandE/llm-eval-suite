import json
from pathlib import Path

import pytest


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
