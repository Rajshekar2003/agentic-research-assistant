"""Tests for the /research endpoint — RAG baseline (mocked search + LLM).

Day 12: 504 timeout test added — baseline pipeline exceeding _BASELINE_TIMEOUT_SECONDS
returns HTTP 504 (distinct from 503 which means a downstream provider is unavailable).
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.llm.client import LLMResult, LLMUnavailableError
from app.main import app
from app.schemas import ResearchResponse
from app.tools.search import SearchResult, SearchUnavailableError

client = TestClient(app)


def _mock_llm_result(text: str = "Mocked LLM answer.") -> LLMResult:
    return LLMResult(
        text=text,
        provider="groq",
        model="llama-3.3-70b-versatile",
        latency_ms=42,
        tokens_in=10,
        tokens_out=20,
    )


def _mock_search_results() -> list[SearchResult]:
    return [
        SearchResult(
            title="Example Source",
            url="https://example.com/article",
            content="Relevant content about the topic.",
            score=0.9,
        )
    ]


def test_research_accepts_valid_query(monkeypatch):
    """Valid query returns 200 with grounded answer, sources populated, mode='baseline'."""
    expected_text = "The capital of France is Paris. [1]"
    mock_complete = AsyncMock(return_value=_mock_llm_result(expected_text))
    mock_llm = MagicMock()
    mock_llm.complete = mock_complete
    monkeypatch.setattr("app.api.research.get_llm_client", lambda: mock_llm)
    monkeypatch.setattr(
        "app.api.research.search",
        AsyncMock(return_value=_mock_search_results()),
    )

    response = client.post("/research", json={"query": "what is the capital of France"})

    assert response.status_code == 200
    data = ResearchResponse(**response.json())
    assert data.answer == expected_text
    assert data.mode == "baseline"
    assert data.elapsed_ms >= 0
    assert len(data.sources) == 1
    assert data.sources[0].title == "Example Source"


def test_research_returns_503_when_llm_unavailable(monkeypatch):
    """LLMUnavailableError is caught and returned as HTTP 503 with correct detail."""
    mock_complete = AsyncMock(side_effect=LLMUnavailableError("Both failed."))
    mock_llm = MagicMock()
    mock_llm.complete = mock_complete
    monkeypatch.setattr("app.api.research.get_llm_client", lambda: mock_llm)
    monkeypatch.setattr(
        "app.api.research.search",
        AsyncMock(return_value=_mock_search_results()),
    )

    response = client.post("/research", json={"query": "what is the capital of France"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Research service temporarily unavailable"


def test_research_returns_503_when_search_unavailable(monkeypatch):
    """SearchUnavailableError short-circuits the pipeline and returns HTTP 503."""
    monkeypatch.setattr(
        "app.api.research.search",
        AsyncMock(
            side_effect=SearchUnavailableError("Search service temporarily unavailable")
        ),
    )

    response = client.post("/research", json={"query": "what is the capital of France"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Search service temporarily unavailable"


def test_research_rejects_short_query():
    """Query shorter than 3 characters returns 422."""
    response = client.post("/research", json={"query": "hi"})
    assert response.status_code == 422


def test_research_rejects_whitespace_only():
    """Query consisting only of whitespace/punctuation returns 422."""
    response = client.post("/research", json={"query": "   ???   "})
    assert response.status_code == 422


def test_baseline_endpoint_returns_504_on_overall_timeout(monkeypatch):
    """When the search + LLM pipeline runs past _BASELINE_TIMEOUT_SECONDS, returns HTTP 504.

    504 (Gateway Timeout) is distinct from 503 (provider unavailable) — it signals
    that our own pipeline ran too long, not that a downstream service is down.
    """

    async def _slow_search(query, max_results):
        await asyncio.sleep(5.0)
        return []

    monkeypatch.setattr("app.api.research.search", _slow_search)
    monkeypatch.setattr("app.api.research._BASELINE_TIMEOUT_SECONDS", 0.1)

    response = client.post("/research", json={"query": "timeout test query"})

    assert response.status_code == 504
    assert "took too long" in response.json()["detail"]
