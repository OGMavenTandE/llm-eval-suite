def test_artifacts_endpoint_for_dry_run_includes_audit_file(client, api_workspace):
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

    response = client.get(f"/runs/{run_id}/artifacts")
    assert response.status_code == 200
    payload = response.json()
    assert payload["run_id"] == run_id
    kinds = {item["kind"] for item in payload["files"]}
    assert "audit" in kinds
