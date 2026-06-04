from apps.api.schemas.common import ErrorResponse


def test_create_run_with_inline_config_validates_and_returns_structured_response(client, api_workspace):
    dataset_path = api_workspace.datasets_dir / "sample.jsonl"
    response = client.post(
        "/runs",
        json={
            "dry_run": True,
            "run_name": "api-dry-run",
            "output_dir": str(api_workspace.output_dir),
            "config": {
                "run_name": "api-dry-run",
                "output_dir": str(api_workspace.output_dir),
                "dataset": str(dataset_path),
                "models": [{"name": "test-model", "provider": "ollama"}],
                "evaluators": [{"name": "correctness", "mode": "exact_match", "threshold": 0.8}],
            },
        },
    )
    assert response.status_code == 202
    payload = response.json()
    assert payload["run_id"]
    assert payload["status"] in {"pending", "running", "validated", "failed_validation", "failed_runtime"}
    assert payload["created_at"]
    assert payload["output_dir"] == str(api_workspace.output_dir)
    assert payload["dry_run"] is True
    assert any(link["rel"] == "results" for link in payload["links"])


def test_create_run_requires_config(client):
    response = client.post("/runs", json={"dry_run": True})
    assert response.status_code == 400
    payload = response.json()
    assert payload["error"] == "bad_request"
    assert payload["message"]


def test_list_runs_returns_structured_list(client, api_workspace):
    dataset_path = api_workspace.datasets_dir / "sample.jsonl"
    create_response = client.post(
        "/runs",
        json={
            "dry_run": True,
            "output_dir": str(api_workspace.output_dir),
            "config": {
                "dataset": str(dataset_path),
                "models": [{"name": "test-model", "provider": "ollama"}],
                "evaluators": [{"name": "correctness", "mode": "exact_match", "threshold": 0.8}],
            },
        },
    )
    run_id = create_response.json()["run_id"]

    from tests.api.conftest import wait_for_terminal_status

    wait_for_terminal_status(client, run_id)

    list_response = client.get("/runs")
    assert list_response.status_code == 200
    payload = list_response.json()
    assert payload["count"] >= 1
    assert any(run["run_id"] == run_id for run in payload["runs"])


def test_run_detail_results_audit_and_artifacts(client, api_workspace):
    dataset_path = api_workspace.datasets_dir / "sample.jsonl"
    create_response = client.post(
        "/runs",
        json={
            "dry_run": True,
            "output_dir": str(api_workspace.output_dir),
            "config": {
                "dataset": str(dataset_path),
                "models": [{"name": "test-model", "provider": "ollama"}],
                "evaluators": [{"name": "correctness", "mode": "exact_match", "threshold": 0.8}],
            },
        },
    )
    run_id = create_response.json()["run_id"]

    from tests.api.conftest import wait_for_terminal_status

    detail = wait_for_terminal_status(client, run_id)
    assert detail["run_id"] == run_id
    assert detail["status"] == "validated"

    detail_response = client.get(f"/runs/{run_id}")
    assert detail_response.status_code == 200

    results_response = client.get(f"/runs/{run_id}/results")
    assert results_response.status_code == 200
    assert results_response.json()["run_id"] == run_id

    audit_response = client.get(f"/runs/{run_id}/audit")
    assert audit_response.status_code == 200
    audit_payload = audit_response.json()
    assert audit_payload["run_id"] == run_id
    assert audit_payload["audit"]["run_id"] == run_id
    assert audit_payload["audit"]["dataset_sample_count"] == 1

    artifacts_response = client.get(f"/runs/{run_id}/artifacts")
    assert artifacts_response.status_code == 200
    assert artifacts_response.json()["run_id"] == run_id


def test_missing_run_returns_standard_error(client):
    response = client.get("/runs/does-not-exist")
    assert response.status_code == 404
    payload = response.json()
    ErrorResponse.model_validate(payload)
    assert payload["error"] == "not_found"
