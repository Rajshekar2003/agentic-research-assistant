from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_includes_version():
    response = client.get("/health")
    data = response.json()
    assert "version" in data
    assert isinstance(data["version"], str)
    assert len(data["version"]) > 0


def test_health_includes_env():
    response = client.get("/health")
    data = response.json()
    assert "env" in data
    assert isinstance(data["env"], str)
    assert data["env"] in ("development", "staging", "production")
