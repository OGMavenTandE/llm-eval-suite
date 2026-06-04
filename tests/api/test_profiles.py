def test_profiles_returns_expected_shape(client):
    response = client.get("/profiles")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] >= 1
    assert len(payload["profiles"]) >= 1

    profile = payload["profiles"][0]
    assert profile["profile_id"]
    assert profile["name"]
    assert profile["path"]
    assert isinstance(profile["evaluators"], list)
    assert isinstance(profile["model_names"], list)
    assert isinstance(profile["available"], bool)
    assert isinstance(profile["valid"], bool)
