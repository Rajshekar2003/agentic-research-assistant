"""Tests for the POST /research/graph endpoint (LangGraph path).

Day 10: 4-node graph (planner → searcher → fact_checker → writer).
Searcher does not call the LLM. FactChecker is verification-only.
Total LLM calls per request: 3 (planner + fact_checker + writer).
"""

import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.llm.client import LLMResult, LLMUnavailableError
from app.main import app
from app.schemas import ResearchResponse
from app.tools.search import SearchResult, SearchUnavailableError

client = TestClient(app)

_WRITER_FALLBACK = "I couldn't verify any claims from the available sources for this question."


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


def _fc_facts_json() -> str:
    """Return a valid FactChecker JSON response with one fact (no 'answer' key — Day 10)."""
    return json.dumps(
        {
            "facts": [{"claim": "France's capital is Paris.", "sources": [1]}],
        }
    )


def _fc_empty_facts_json() -> str:
    """Return a valid FactChecker JSON response with an empty facts list."""
    return json.dumps({"facts": []})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_graph_endpoint_happy_path(monkeypatch):
    """Valid query returns 200 with mode='graph', grounded answer, sources, and facts."""
    expected_text = "France's capital is Paris. [1]"

    planner_mock = MagicMock()
    planner_mock.complete = AsyncMock(
        return_value=_mock_llm_result('["What is the capital of France?"]')
    )
    fc_mock = MagicMock()
    fc_mock.complete = AsyncMock(return_value=_mock_llm_result(_fc_facts_json()))

    writer_mock = MagicMock()
    writer_mock.complete = AsyncMock(return_value=_mock_llm_result(expected_text))

    monkeypatch.setattr("app.agents.planner.get_llm_client", lambda: planner_mock)
    monkeypatch.setattr(
        "app.agents.searcher.search", AsyncMock(return_value=_mock_search_results())
    )
    monkeypatch.setattr("app.agents.fact_checker.get_llm_client", lambda: fc_mock)
    monkeypatch.setattr("app.agents.writer.get_llm_client", lambda: writer_mock)

    response = client.post("/research/graph", json={"query": "what is the capital of France"})

    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "graph"
    assert data["answer"] == expected_text
    assert data["elapsed_ms"] >= 0
    assert len(data["sources"]) == 1
    assert data["sources"][0]["title"] == "Graph Endpoint Source"
    assert data["facts"] is not None
    assert len(data["facts"]) == 1
    assert data["facts"][0]["claim"] == "France's capital is Paris."
    assert data["facts"][0]["sources"] == [1]


def test_graph_endpoint_returns_503_on_search_failure(monkeypatch):
    """SearchUnavailableError from the searcher node is caught and returned as HTTP 503."""
    plan_mock = MagicMock()
    plan_mock.complete = AsyncMock(
        return_value=_mock_llm_result('["what is the capital of France"]')
    )
    monkeypatch.setattr("app.agents.planner.get_llm_client", lambda: plan_mock)
    monkeypatch.setattr(
        "app.agents.searcher.search",
        AsyncMock(side_effect=SearchUnavailableError("Search service temporarily unavailable")),
    )

    response = client.post("/research/graph", json={"query": "what is the capital of France"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Search service temporarily unavailable"


def test_graph_endpoint_returns_503_on_llm_failure(monkeypatch):
    """LLMUnavailableError from the planner node is caught and returned as HTTP 503."""
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(side_effect=LLMUnavailableError("Both providers failed."))
    monkeypatch.setattr("app.agents.planner.get_llm_client", lambda: mock_llm)

    response = client.post("/research/graph", json={"query": "what is the capital of France"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Research service temporarily unavailable"


def test_graph_endpoint_returns_503_on_fact_checker_llm_failure(monkeypatch):
    """LLMUnavailableError from fact_checker is caught and returned as HTTP 503."""
    plan_mock = MagicMock()
    plan_mock.complete = AsyncMock(
        return_value=_mock_llm_result('["what is the capital of France"]')
    )
    fc_mock = MagicMock()
    fc_mock.complete = AsyncMock(side_effect=LLMUnavailableError("Both providers failed."))

    monkeypatch.setattr("app.agents.planner.get_llm_client", lambda: plan_mock)
    monkeypatch.setattr(
        "app.agents.searcher.search", AsyncMock(return_value=_mock_search_results())
    )
    monkeypatch.setattr("app.agents.fact_checker.get_llm_client", lambda: fc_mock)

    response = client.post("/research/graph", json={"query": "what is the capital of France"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Research service temporarily unavailable"


def test_graph_endpoint_returns_503_when_writer_llm_fails(monkeypatch):
    """LLMUnavailableError from the writer node is caught and returned as HTTP 503."""
    plan_mock = MagicMock()
    plan_mock.complete = AsyncMock(
        return_value=_mock_llm_result('["what is the capital of France"]')
    )
    fc_mock = MagicMock()
    fc_mock.complete = AsyncMock(return_value=_mock_llm_result(_fc_facts_json()))

    writer_mock = MagicMock()
    writer_mock.complete = AsyncMock(side_effect=LLMUnavailableError("Both providers failed."))

    monkeypatch.setattr("app.agents.planner.get_llm_client", lambda: plan_mock)
    monkeypatch.setattr(
        "app.agents.searcher.search", AsyncMock(return_value=_mock_search_results())
    )
    monkeypatch.setattr("app.agents.fact_checker.get_llm_client", lambda: fc_mock)
    monkeypatch.setattr("app.agents.writer.get_llm_client", lambda: writer_mock)

    response = client.post("/research/graph", json={"query": "what is the capital of France"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Research service temporarily unavailable"


def test_graph_endpoint_writer_skipped_when_no_facts(monkeypatch):
    """FactChecker returns empty facts → Writer skips LLM, returns fallback answer, status 200.

    This is intentional graceful behaviour: an unanswerable query is not an error.
    """
    plan_mock = MagicMock()
    plan_mock.complete = AsyncMock(
        return_value=_mock_llm_result('["what is the capital of France"]')
    )
    fc_mock = MagicMock()
    fc_mock.complete = AsyncMock(return_value=_mock_llm_result(_fc_empty_facts_json()))

    writer_llm_call_count = [0]

    def counting_writer_llm():
        writer_llm_call_count[0] += 1
        raise AssertionError("Writer must not call LLM when facts are empty")

    monkeypatch.setattr("app.agents.planner.get_llm_client", lambda: plan_mock)
    monkeypatch.setattr(
        "app.agents.searcher.search", AsyncMock(return_value=_mock_search_results())
    )
    monkeypatch.setattr("app.agents.fact_checker.get_llm_client", lambda: fc_mock)
    monkeypatch.setattr("app.agents.writer.get_llm_client", counting_writer_llm)

    response = client.post("/research/graph", json={"query": "what is the capital of France"})

    assert response.status_code == 200
    assert response.json()["answer"] == _WRITER_FALLBACK
    assert writer_llm_call_count[0] == 0


def test_graph_endpoint_rejects_short_query():
    """Query shorter than 3 characters returns 422 (validation enforced before graph runs)."""
    response = client.post("/research/graph", json={"query": "hi"})
    assert response.status_code == 422


def test_graph_endpoint_logs_plan_size_and_facts_count(monkeypatch, caplog):
    """Successful graph request emits a log line with plan_size=N and facts_count=N."""
    plan_json = '["What is the capital?", "Where is France located?"]'
    expected_answer = "Paris is the capital of France. [1]"

    planner_mock = MagicMock()
    planner_mock.complete = AsyncMock(return_value=_mock_llm_result(plan_json))
    fc_mock = MagicMock()
    fc_mock.complete = AsyncMock(return_value=_mock_llm_result(_fc_facts_json()))

    writer_mock = MagicMock()
    writer_mock.complete = AsyncMock(return_value=_mock_llm_result(expected_answer))

    monkeypatch.setattr("app.agents.planner.get_llm_client", lambda: planner_mock)
    monkeypatch.setattr(
        "app.agents.searcher.search", AsyncMock(return_value=_mock_search_results())
    )
    monkeypatch.setattr("app.agents.fact_checker.get_llm_client", lambda: fc_mock)
    monkeypatch.setattr("app.agents.writer.get_llm_client", lambda: writer_mock)

    with caplog.at_level(logging.INFO, logger="app.api.research"):
        response = client.post("/research/graph", json={"query": "what is the capital of France"})

    assert response.status_code == 200
    log_messages = " ".join(r.message for r in caplog.records)
    assert "plan_size=2" in log_messages
    assert "facts_count=1" in log_messages
