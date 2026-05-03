from fastapi.testclient import TestClient

from app.main import app
from app.schemas import ResearchResponse

client = TestClient(app)


def test_research_accepts_valid_query():
    response = client.post("/research", json={"query": "what is the capital of France"})
    assert response.status_code == 200
    data = response.json()
    validated = ResearchResponse(**data)
    assert validated.elapsed_ms >= 0
    assert validated.mode == "baseline"


def test_research_rejects_short_query():
    response = client.post("/research", json={"query": "hi"})
    assert response.status_code == 422


def test_research_rejects_whitespace_only():
    response = client.post("/research", json={"query": "   ???   "})
    assert response.status_code == 422
