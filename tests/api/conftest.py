import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apps.api.dependencies import AppSettings, override_settings, reset_run_job_manager, reset_settings
from apps.api.main import create_app


@pytest.fixture
def api_workspace(tmp_path: Path):
    config_dir = tmp_path / "config"
    datasets_dir = tmp_path / "datasets"
    output_dir = tmp_path / "results"
    config_dir.mkdir()
    datasets_dir.mkdir()
    output_dir.mkdir()

    dataset_path = datasets_dir / "sample.jsonl"
    dataset_path.write_text(
        '{"prompt": "What is 2+2?", "expected_answer": "4"}\n',
        encoding="utf-8",
    )

    config_path = config_dir / "test_profile.yaml"
    config_path.write_text(
        """
run_name: test-run
output_dir: RESULTS_DIR
dataset: DATASET_PATH
models:
  - name: test-model
    provider: ollama
evaluators:
  - name: correctness
    mode: exact_match
    threshold: 0.8
""".replace(
            "RESULTS_DIR", str(output_dir)
        ).replace(
            "DATASET_PATH", str(dataset_path)
        ),
        encoding="utf-8",
    )

    settings = AppSettings(
        workspace_root=tmp_path,
        output_dir=output_dir,
        config_dir=config_dir,
        datasets_dir=datasets_dir,
    )
    override_settings(settings)
    reset_run_job_manager()
    yield settings
    reset_settings()
    reset_run_job_manager()


@pytest.fixture
def client(api_workspace: AppSettings):
    return TestClient(create_app())


def wait_for_terminal_status(client: TestClient, run_id: str, timeout_seconds: float = 5.0) -> dict:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        response = client.get(f"/runs/{run_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] not in {"pending", "running"}:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"Run {run_id} did not finish within {timeout_seconds} seconds")
