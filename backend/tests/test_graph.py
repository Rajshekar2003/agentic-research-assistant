"""Tests for the LangGraph workflow — compiled graph and 4-node pipeline behaviour."""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.graph.workflow import get_compiled_graph
from app.llm.client import LLMResult, LLMUnavailableError
from app.tools.search import SearchResult, SearchUnavailableError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_llm_result(text: str = "Graph LLM answer.") -> LLMResult:
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
            title="Graph Source",
            url="https://example.com/graph-article",
            content="Content relevant to the graph test query.",
            score=0.95,
        )
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_graph_compiles():
    """get_compiled_graph() returns a compiled graph and is a true singleton."""
    g1 = get_compiled_graph()
    g2 = get_compiled_graph()
    assert g1 is not None
    assert g1 is g2


async def test_graph_four_node_happy_path(monkeypatch):
    """Planner + Searcher + FactChecker + Writer nodes together populate all expected state fields."""
    expected_answer = "The answer is 42. [1]"
    fc_response = json.dumps(
        {
            "facts": [{"claim": "42 is the answer.", "sources": [1]}],
        }
    )

    planner_mock = MagicMock()
    planner_mock.complete = AsyncMock(return_value=_mock_llm_result('["What is 42?"]'))

    fc_mock = MagicMock()
    fc_mock.complete = AsyncMock(return_value=_mock_llm_result(fc_response))

    writer_mock = MagicMock()
    writer_mock.complete = AsyncMock(return_value=_mock_llm_result(expected_answer))

    monkeypatch.setattr("app.agents.planner.get_llm_client", lambda: planner_mock)
    monkeypatch.setattr(
        "app.agents.searcher.search", AsyncMock(return_value=_mock_search_results())
    )
    monkeypatch.setattr("app.agents.fact_checker.get_llm_client", lambda: fc_mock)
    monkeypatch.setattr("app.agents.writer.get_llm_client", lambda: writer_mock)

    graph = get_compiled_graph()
    state = await graph.ainvoke({"query": "test query for graph"})

    assert state["final_answer"] == expected_answer
    assert len(state["sources"]) == 1
    assert state["sources"][0].title == "Graph Source"
    assert state["provider"] == "groq"
    assert state["model"] == "llama-3.3-70b-versatile"
    assert state["tokens_in"] == 10
    assert state["tokens_out"] == 20
    assert state["plan"] == ["What is 42?"]
    assert len(state["search_results"]) == 1
    assert len(state["facts"]) == 1
    assert state["facts"][0]["claim"] == "42 is the answer."


async def test_graph_propagates_search_failure(monkeypatch):
    """SearchUnavailableError raised inside the searcher propagates through ainvoke().

    The planner succeeds first; then the searcher's Tavily call fails.  The exception
    propagates out of ainvoke() so the API endpoint can convert it to a 503.
    """
    plan_llm = MagicMock()
    plan_llm.complete = AsyncMock(return_value=_mock_llm_result('["test question"]'))
    monkeypatch.setattr("app.agents.planner.get_llm_client", lambda: plan_llm)
    monkeypatch.setattr(
        "app.agents.searcher.search",
        AsyncMock(side_effect=SearchUnavailableError("Search service temporarily unavailable")),
    )

    graph = get_compiled_graph()
    with pytest.raises(SearchUnavailableError):
        await graph.ainvoke({"query": "test query"})


async def test_graph_propagates_llm_failure(monkeypatch):
    """LLMUnavailableError raised inside the planner propagates through ainvoke().

    Same propagation contract as search failures — the API endpoint catches it.
    """
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(side_effect=LLMUnavailableError("Both providers failed."))
    monkeypatch.setattr("app.agents.planner.get_llm_client", lambda: mock_llm)

    graph = get_compiled_graph()
    with pytest.raises(LLMUnavailableError):
        await graph.ainvoke({"query": "test query"})
