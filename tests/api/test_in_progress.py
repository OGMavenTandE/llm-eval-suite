def test_in_progress_results_return_structured_not_error(client, api_workspace, monkeypatch):
    """Results endpoint should return ready=false while a run is in progress."""
    from apps.api.services import run_jobs as run_jobs_module

    original_execute = run_jobs_module.RunJobManager._execute

    def slow_execute(self, job, config, dry_run, compare):
        import time

        job.status = "running"
        time.sleep(2)
        return original_execute(self, job, config, dry_run, compare)

    monkeypatch.setattr(run_jobs_module.RunJobManager, "_execute", slow_execute)

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

    results_response = client.get(f"/runs/{run_id}/results")
    assert results_response.status_code == 200
    payload = results_response.json()
    assert payload["ready"] is False
    assert payload["status"] in {"pending", "running"}
    assert payload["message"]
