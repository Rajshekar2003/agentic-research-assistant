"""Tests for the Searcher agent (app/agents/searcher.py)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.searcher import run as searcher_run
from app.llm.client import LLMResult
from app.tools.search import SearchResult, SearchUnavailableError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_search_result(url: str, score: float = 0.9) -> SearchResult:
    return SearchResult(
        title=f"Title for {url}",
        url=url,
        content=f"Content from {url}",
        score=score,
    )


def _mock_llm_result(text: str = "Synthesized answer.") -> LLMResult:
    return LLMResult(
        text=text,
        provider="groq",
        model="llama-3.3-70b-versatile",
        latency_ms=60,
        tokens_in=100,
        tokens_out=30,
    )


def _make_mock_llm(text: str = "Synthesized answer.") -> MagicMock:
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(return_value=_mock_llm_result(text))
    return mock_llm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_searcher_runs_search_per_subquestion(monkeypatch):
    """Searcher calls search() exactly once per sub-question with the correct query string."""
    plan = ["cause of WW1", "timeline of WW1", "key leaders of WW1"]
    mock_search = AsyncMock(return_value=[_mock_search_result("https://example.com/1")])
    monkeypatch.setattr("app.agents.searcher.search", mock_search)
    monkeypatch.setattr("app.agents.searcher.get_llm_client", lambda: _make_mock_llm())

    state = {"query": "Tell me about World War 1", "plan": plan}
    await searcher_run(state)

    assert mock_search.call_count == 3
    called_queries = [c.args[0] for c in mock_search.call_args_list]
    assert called_queries == plan


async def test_searcher_dedupes_urls_across_calls(monkeypatch):
    """Searcher removes duplicate URLs that appear in multiple sub-question results."""
    shared_url = "https://shared.example.com/article"
    results_q1 = [
        _mock_search_result(shared_url),
        _mock_search_result("https://unique1.example.com"),
    ]
    results_q2 = [
        _mock_search_result(shared_url),
        _mock_search_result("https://unique2.example.com"),
    ]
    mock_search = AsyncMock(side_effect=[results_q1, results_q2])
    monkeypatch.setattr("app.agents.searcher.search", mock_search)
    monkeypatch.setattr("app.agents.searcher.get_llm_client", lambda: _make_mock_llm())

    state = {"query": "test dedup query", "plan": ["q1", "q2"]}
    result = await searcher_run(state)

    urls = [r["url"] for r in result["search_results"]]
    assert len(urls) == len(set(urls)), "Duplicate URLs found in search_results"
    assert len(urls) == 3  # shared_url + unique1 + unique2


async def test_searcher_caps_results_at_8(monkeypatch):
    """Searcher returns at most 8 results even when sub-questions yield more unique URLs."""

    def _make_results(prefix: str) -> list[SearchResult]:
        return [
            _mock_search_result(f"https://{prefix}-{i}.example.com", score=float(i))
            for i in range(5)
        ]

    # 4 sub-questions × 5 unique URLs each = 20 unique results before the cap.
    mock_search = AsyncMock(
        side_effect=[
            _make_results("a"),
            _make_results("b"),
            _make_results("c"),
            _make_results("d"),
        ]
    )
    monkeypatch.setattr("app.agents.searcher.search", mock_search)
    monkeypatch.setattr("app.agents.searcher.get_llm_client", lambda: _make_mock_llm())

    state = {"query": "complex query", "plan": ["q1", "q2", "q3", "q4"]}
    result = await searcher_run(state)

    assert len(result["search_results"]) == 8


async def test_searcher_handles_single_element_plan(monkeypatch):
    """With a 1-item plan, search is called once with max_results=5."""
    mock_search = AsyncMock(return_value=[_mock_search_result("https://example.com/single")])
    monkeypatch.setattr("app.agents.searcher.search", mock_search)
    monkeypatch.setattr("app.agents.searcher.get_llm_client", lambda: _make_mock_llm())

    state = {"query": "simple question", "plan": ["simple question"]}
    await searcher_run(state)

    assert mock_search.call_count == 1
    assert mock_search.call_args.kwargs["max_results"] == 5


async def test_searcher_propagates_search_failure(monkeypatch):
    """SearchUnavailableError from Tavily propagates through the searcher (not caught)."""
    monkeypatch.setattr(
        "app.agents.searcher.search",
        AsyncMock(side_effect=SearchUnavailableError("Search service temporarily unavailable")),
    )

    state = {"query": "some query", "plan": ["some query"]}
    with pytest.raises(SearchUnavailableError):
        await searcher_run(state)
