"""Tests for the /research endpoint — baseline (real LLM, no search)."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.llm.client import LLMResult, LLMUnavailableError
from app.main import app
from app.schemas import ResearchResponse

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


def test_research_accepts_valid_query(monkeypatch):
    """Valid query returns 200 with model-generated answer and mode='baseline'."""
    expected_text = "The capital of France is Paris."
    mock_complete = AsyncMock(return_value=_mock_llm_result(expected_text))
    mock_llm = MagicMock()
    mock_llm.complete = mock_complete
    monkeypatch.setattr("app.api.research.get_llm_client", lambda: mock_llm)

    response = client.post("/research", json={"query": "what is the capital of France"})

    assert response.status_code == 200
    data = ResearchResponse(**response.json())
    assert data.answer == expected_text
    assert data.mode == "baseline"
    assert data.elapsed_ms >= 0


def test_research_returns_503_when_llm_unavailable(monkeypatch):
    """LLMUnavailableError is caught and returned as HTTP 503."""
    mock_complete = AsyncMock(side_effect=LLMUnavailableError("Both failed."))
    mock_llm = MagicMock()
    mock_llm.complete = mock_complete
    monkeypatch.setattr("app.api.research.get_llm_client", lambda: mock_llm)

    response = client.post("/research", json={"query": "what is the capital of France"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Research service temporarily unavailable"


def test_research_rejects_short_query():
    """Query shorter than 3 characters returns 422."""
    response = client.post("/research", json={"query": "hi"})
    assert response.status_code == 422


def test_research_rejects_whitespace_only():
    """Query consisting only of whitespace/punctuation returns 422."""
    response = client.post("/research", json={"query": "   ???   "})
    assert response.status_code == 422
