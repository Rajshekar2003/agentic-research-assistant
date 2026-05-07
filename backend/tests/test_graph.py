"""Tests for the LangGraph workflow — compiled graph and 5-node pipeline behaviour.

Day 11: graph has 5 nodes (planner, searcher, fact_checker, writer, critic) with a
conditional feedback edge from critic back to writer, capped at 2 revisions.
"""

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


def _fc_facts_json() -> str:
    return json.dumps({"facts": [{"claim": "42 is the answer.", "sources": [1]}]})


def _approve_json() -> str:
    return json.dumps({"verdict": "approve", "critique": ""})


def _revise_json(critique: str = "Claim is not supported; cite source.") -> str:
    return json.dumps({"verdict": "revise", "critique": critique})


# ---------------------------------------------------------------------------
# Tests — graph compilation
# ---------------------------------------------------------------------------


def test_graph_compiles():
    """get_compiled_graph() returns a compiled graph and is a true singleton."""
    g1 = get_compiled_graph()
    g2 = get_compiled_graph()
    assert g1 is not None
    assert g1 is g2


# ---------------------------------------------------------------------------
# Tests — 5-node happy path
# ---------------------------------------------------------------------------


async def test_graph_five_node_happy_path(monkeypatch):
    """All 5 nodes together populate expected state; Critic approves on first pass."""
    expected_answer = "The answer is 42. [1]"

    planner_mock = MagicMock()
    planner_mock.complete = AsyncMock(return_value=_mock_llm_result('["What is 42?"]'))

    fc_mock = MagicMock()
    fc_mock.complete = AsyncMock(return_value=_mock_llm_result(_fc_facts_json()))

    writer_mock = MagicMock()
    writer_mock.complete = AsyncMock(return_value=_mock_llm_result(expected_answer))

    critic_mock = MagicMock()
    critic_mock.complete = AsyncMock(return_value=_mock_llm_result(_approve_json()))

    monkeypatch.setattr("app.agents.planner.get_llm_client", lambda: planner_mock)
    monkeypatch.setattr(
        "app.agents.searcher.search", AsyncMock(return_value=_mock_search_results())
    )
    monkeypatch.setattr("app.agents.fact_checker.get_llm_client", lambda: fc_mock)
    monkeypatch.setattr("app.agents.writer.get_llm_client", lambda: writer_mock)
    monkeypatch.setattr("app.agents.critic.get_llm_client", lambda: critic_mock)

    graph = get_compiled_graph()
    state = await graph.ainvoke({"query": "test query for graph"})

    assert state["final_answer"] == expected_answer
    assert len(state["sources"]) == 1
    assert state["sources"][0].title == "Graph Source"
    assert state["provider"] == "groq"
    assert state["plan"] == ["What is 42?"]
    assert len(state["facts"]) == 1
    assert state["critic_verdict"] == "approve"
    assert state.get("revision_count", 0) == 0

    # 4 LLM calls: planner + fact_checker + writer + critic
    assert planner_mock.complete.call_count == 1
    assert fc_mock.complete.call_count == 1
    assert writer_mock.complete.call_count == 1
    assert critic_mock.complete.call_count == 1


# ---------------------------------------------------------------------------
# Tests — Critic feedback loop
# ---------------------------------------------------------------------------


async def test_graph_no_revision_when_critic_approves_first(monkeypatch):
    """Critic approves on first call → Writer called once, Critic called once, revision_count=0."""
    planner_mock = MagicMock()
    planner_mock.complete = AsyncMock(return_value=_mock_llm_result('["test question"]'))

    fc_mock = MagicMock()
    fc_mock.complete = AsyncMock(return_value=_mock_llm_result(_fc_facts_json()))

    writer_mock = MagicMock()
    writer_mock.complete = AsyncMock(return_value=_mock_llm_result("First and only draft."))

    critic_mock = MagicMock()
    critic_mock.complete = AsyncMock(return_value=_mock_llm_result(_approve_json()))

    monkeypatch.setattr("app.agents.planner.get_llm_client", lambda: planner_mock)
    monkeypatch.setattr(
        "app.agents.searcher.search", AsyncMock(return_value=_mock_search_results())
    )
    monkeypatch.setattr("app.agents.fact_checker.get_llm_client", lambda: fc_mock)
    monkeypatch.setattr("app.agents.writer.get_llm_client", lambda: writer_mock)
    monkeypatch.setattr("app.agents.critic.get_llm_client", lambda: critic_mock)

    state = await get_compiled_graph().ainvoke({"query": "test query"})

    assert state["critic_verdict"] == "approve"
    assert state.get("revision_count", 0) == 0
    assert writer_mock.complete.call_count == 1
    assert critic_mock.complete.call_count == 1


async def test_graph_revision_loop_runs_once(monkeypatch):
    """Critic revises once then approves: Writer×2, Critic×2, revision_count=1."""
    planner_mock = MagicMock()
    planner_mock.complete = AsyncMock(return_value=_mock_llm_result('["test question"]'))

    fc_mock = MagicMock()
    fc_mock.complete = AsyncMock(return_value=_mock_llm_result(_fc_facts_json()))

    writer_mock = MagicMock()
    writer_mock.complete = AsyncMock(
        side_effect=[
            _mock_llm_result("First draft."),
            _mock_llm_result("Revised draft."),
        ]
    )

    critic_mock = MagicMock()
    critic_mock.complete = AsyncMock(
        side_effect=[
            _mock_llm_result(_revise_json("Remove unsupported claim.")),
            _mock_llm_result(_approve_json()),
        ]
    )

    monkeypatch.setattr("app.agents.planner.get_llm_client", lambda: planner_mock)
    monkeypatch.setattr(
        "app.agents.searcher.search", AsyncMock(return_value=_mock_search_results())
    )
    monkeypatch.setattr("app.agents.fact_checker.get_llm_client", lambda: fc_mock)
    monkeypatch.setattr("app.agents.writer.get_llm_client", lambda: writer_mock)
    monkeypatch.setattr("app.agents.critic.get_llm_client", lambda: critic_mock)

    state = await get_compiled_graph().ainvoke({"query": "test query"})

    assert state["critic_verdict"] == "approve"
    assert state["revision_count"] == 1
    assert state["final_answer"] == "Revised draft."
    assert writer_mock.complete.call_count == 2
    assert critic_mock.complete.call_count == 2


async def test_graph_revision_loop_hits_cap(monkeypatch):
    """Critic keeps saying revise; cap stops after revision_count reaches 2.

    With route `revision_count < 2`: Writer runs initial + 1 revision = 2 times total,
    Critic runs 2 times.  The cap fires when revision_count==2 (route returns END).
    final critic_verdict="revise" (cap was hit, not an approval).
    """
    planner_mock = MagicMock()
    planner_mock.complete = AsyncMock(return_value=_mock_llm_result('["test question"]'))

    fc_mock = MagicMock()
    fc_mock.complete = AsyncMock(return_value=_mock_llm_result(_fc_facts_json()))

    writer_mock = MagicMock()
    writer_mock.complete = AsyncMock(
        side_effect=[
            _mock_llm_result("First draft."),
            _mock_llm_result("Revised draft — cap hit."),
        ]
    )

    # Always says revise — cap terminates after 2 revise verdicts (revision_count==2)
    critic_mock = MagicMock()
    critic_mock.complete = AsyncMock(
        side_effect=[
            _mock_llm_result(_revise_json("Fix claim A.")),
            _mock_llm_result(_revise_json("Fix claim B.")),
        ]
    )

    monkeypatch.setattr("app.agents.planner.get_llm_client", lambda: planner_mock)
    monkeypatch.setattr(
        "app.agents.searcher.search", AsyncMock(return_value=_mock_search_results())
    )
    monkeypatch.setattr("app.agents.fact_checker.get_llm_client", lambda: fc_mock)
    monkeypatch.setattr("app.agents.writer.get_llm_client", lambda: writer_mock)
    monkeypatch.setattr("app.agents.critic.get_llm_client", lambda: critic_mock)

    state = await get_compiled_graph().ainvoke({"query": "test query"})

    assert state["revision_count"] == 2
    assert state["critic_verdict"] == "revise"  # hit the cap, never approved
    assert state["final_answer"] == "Revised draft — cap hit."
    assert writer_mock.complete.call_count == 2
    assert critic_mock.complete.call_count == 2


# ---------------------------------------------------------------------------
# Tests — error propagation
# ---------------------------------------------------------------------------


async def test_graph_propagates_search_failure(monkeypatch):
    """SearchUnavailableError raised inside the searcher propagates through ainvoke()."""
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
    """LLMUnavailableError raised inside the planner propagates through ainvoke()."""
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(side_effect=LLMUnavailableError("Both providers failed."))
    monkeypatch.setattr("app.agents.planner.get_llm_client", lambda: mock_llm)

    graph = get_compiled_graph()
    with pytest.raises(LLMUnavailableError):
        await graph.ainvoke({"query": "test query"})


async def test_graph_propagates_critic_llm_failure(monkeypatch):
    """LLMUnavailableError raised inside the critic propagates through ainvoke()."""
    planner_mock = MagicMock()
    planner_mock.complete = AsyncMock(return_value=_mock_llm_result('["test question"]'))

    fc_mock = MagicMock()
    fc_mock.complete = AsyncMock(return_value=_mock_llm_result(_fc_facts_json()))

    writer_mock = MagicMock()
    writer_mock.complete = AsyncMock(return_value=_mock_llm_result("A draft answer."))

    critic_llm = MagicMock()
    critic_llm.complete = AsyncMock(side_effect=LLMUnavailableError("Both providers failed."))

    monkeypatch.setattr("app.agents.planner.get_llm_client", lambda: planner_mock)
    monkeypatch.setattr(
        "app.agents.searcher.search", AsyncMock(return_value=_mock_search_results())
    )
    monkeypatch.setattr("app.agents.fact_checker.get_llm_client", lambda: fc_mock)
    monkeypatch.setattr("app.agents.writer.get_llm_client", lambda: writer_mock)
    monkeypatch.setattr("app.agents.critic.get_llm_client", lambda: critic_llm)

    graph = get_compiled_graph()
    with pytest.raises(LLMUnavailableError):
        await graph.ainvoke({"query": "test query"})
