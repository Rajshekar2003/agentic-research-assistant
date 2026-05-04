"""Tests for the POST /research/graph endpoint (LangGraph path)."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.llm.client import LLMResult, LLMUnavailableError
from app.main import app
from app.schemas import ResearchResponse
from app.tools.search import SearchResult, SearchUnavailableError

client = TestClient(app)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_llm_result(text: str = "Graph endpoint answer.") -> LLMResult:
    return LLMResult(
        text=text,
        provider="groq",
        model="llama-3.3-70b-versatile",
        latency_ms=55,
        tokens_in=12,
        tokens_out=25,
    )


def _mock_search_results() -> list[SearchResult]:
    return [
        SearchResult(
            title="Graph Endpoint Source",
            url="https://example.com/graph-endpoint-article",
            content="Content for the graph endpoint test.",
            score=0.88,
        )
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_graph_endpoint_happy_path(monkeypatch):
    """Valid query returns 200 with mode='graph', grounded answer, and sources."""
    expected_text = "France's capital is Paris. [1]"
    mock_complete = AsyncMock(return_value=_mock_llm_result(expected_text))
    mock_llm = MagicMock()
    mock_llm.complete = mock_complete
    monkeypatch.setattr("app.graph.workflow.search", AsyncMock(return_value=_mock_search_results()))
    monkeypatch.setattr("app.graph.workflow.get_llm_client", lambda: mock_llm)

    response = client.post("/research/graph", json={"query": "what is the capital of France"})

    assert response.status_code == 200
    data = ResearchResponse(**response.json())
    assert data.mode == "graph"
    assert data.answer == expected_text
    assert data.elapsed_ms >= 0
    assert len(data.sources) == 1
    assert data.sources[0].title == "Graph Endpoint Source"


def test_graph_endpoint_returns_503_on_search_failure(monkeypatch):
    """SearchUnavailableError from the graph node is caught and returned as HTTP 503."""
    monkeypatch.setattr(
        "app.graph.workflow.search",
        AsyncMock(
            side_effect=SearchUnavailableError("Search service temporarily unavailable")
        ),
    )

    response = client.post("/research/graph", json={"query": "what is the capital of France"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Search service temporarily unavailable"


def test_graph_endpoint_returns_503_on_llm_failure(monkeypatch):
    """LLMUnavailableError from the graph node is caught and returned as HTTP 503."""
    monkeypatch.setattr(
        "app.graph.workflow.search",
        AsyncMock(return_value=_mock_search_results()),
    )
    mock_complete = AsyncMock(side_effect=LLMUnavailableError("Both providers failed."))
    mock_llm = MagicMock()
    mock_llm.complete = mock_complete
    monkeypatch.setattr("app.graph.workflow.get_llm_client", lambda: mock_llm)

    response = client.post("/research/graph", json={"query": "what is the capital of France"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Research service temporarily unavailable"


def test_graph_endpoint_rejects_short_query():
    """Query shorter than 3 characters returns 422 (validation enforced before graph runs)."""
    response = client.post("/research/graph", json={"query": "hi"})
    assert response.status_code == 422
