def test_health_returns_expected_structure(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["app_status"] == "ok"
    assert payload["api_status"] == "ok"
    assert payload["package_version"]
    assert payload["timestamp"]


def test_system_status_returns_expected_structure(client):
    response = client.get("/system/status")
    assert response.status_code == 200
    payload = response.json()
    assert payload["app_version"]
    assert payload["output_dir"]
    assert payload["run_index_path"]
    assert isinstance(payload["profiles_count"], int)
    assert isinstance(payload["datasets_count"], int)
    assert isinstance(payload["model_providers"], list)
    assert isinstance(payload["warnings"], list)
