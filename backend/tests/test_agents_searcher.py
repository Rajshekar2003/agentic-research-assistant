"""Tests for the Searcher agent (app/agents/searcher.py).

Day 9: Searcher is retrieval-only — it no longer calls the LLM. Tests reflect this:
LLM helpers are removed and test_searcher_does_not_call_llm verifies the invariant.

Day 12: Parallel execution tests verify that sub-question Tavily calls run concurrently
(wall-clock timing), that a single search failure propagates through gather, and that the
20s overall gather timeout raises SearchUnavailableError.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.searcher import run as searcher_run
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_searcher_does_not_call_llm(monkeypatch):
    """Searcher is retrieval-only as of Day 9 — get_llm_client must never be invoked."""
    mock_search = AsyncMock(return_value=[_mock_search_result("https://example.com/1")])
    monkeypatch.setattr("app.agents.searcher.search", mock_search)

    def _llm_must_not_be_called():
        raise AssertionError("Searcher called get_llm_client — it should be retrieval-only")

    monkeypatch.setattr("app.llm.client.get_llm_client", _llm_must_not_be_called)

    state = {"query": "no llm please", "plan": ["no llm please"]}
    result = await searcher_run(state)

    assert result["provider"] is None
    assert result["model"] is None
    assert result["tokens_in"] is None
    assert result["tokens_out"] is None


async def test_searcher_runs_search_per_subquestion(monkeypatch):
    """Searcher calls search() exactly once per sub-question with the correct query string."""
    plan = ["cause of WW1", "timeline of WW1", "key leaders of WW1"]
    mock_search = AsyncMock(return_value=[_mock_search_result("https://example.com/1")])
    monkeypatch.setattr("app.agents.searcher.search", mock_search)

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

    state = {"query": "complex query", "plan": ["q1", "q2", "q3", "q4"]}
    result = await searcher_run(state)

    assert len(result["search_results"]) == 8


async def test_searcher_handles_single_element_plan(monkeypatch):
    """With a 1-item plan, search is called once with max_results=5."""
    mock_search = AsyncMock(return_value=[_mock_search_result("https://example.com/single")])
    monkeypatch.setattr("app.agents.searcher.search", mock_search)

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


# ---------------------------------------------------------------------------
# Day 12: Parallelisation tests
# ---------------------------------------------------------------------------


async def test_searcher_runs_calls_concurrently_not_sequentially(monkeypatch):
    """3 sub-questions each sleeping 0.5s complete in <1.0s total (parallel, not ~1.5s sequential)."""

    async def _slow_search(query, *, max_results):
        await asyncio.sleep(0.5)
        return [_mock_search_result(f"https://example.com/{query.replace(' ', '-')}")]

    monkeypatch.setattr("app.agents.searcher.search", _slow_search)

    state = {"query": "parallelism test", "plan": ["q1", "q2", "q3"]}
    t0 = asyncio.get_event_loop().time()
    await searcher_run(state)
    elapsed = asyncio.get_event_loop().time() - t0

    assert elapsed < 1.0, f"Expected parallel execution (<1.0s), got {elapsed:.2f}s"


async def test_searcher_propagates_first_search_failure_in_gather(monkeypatch):
    """SearchUnavailableError on the second of 3 concurrent calls propagates out of gather."""
    results_q1 = [_mock_search_result("https://example.com/1")]
    results_q3 = [_mock_search_result("https://example.com/3")]

    mock_search = AsyncMock(
        side_effect=[
            results_q1,
            SearchUnavailableError("Tavily failed on second call"),
            results_q3,
        ]
    )
    monkeypatch.setattr("app.agents.searcher.search", mock_search)

    state = {"query": "failure propagation test", "plan": ["q1", "q2", "q3"]}
    with pytest.raises(SearchUnavailableError):
        await searcher_run(state)


async def test_searcher_overall_timeout(monkeypatch):
    """asyncio.wait_for fires when all searches hang, raising SearchUnavailableError."""

    async def _hanging_search(query, *, max_results):
        await asyncio.sleep(5.0)  # longer than the patched timeout
        return []

    monkeypatch.setattr("app.agents.searcher.search", _hanging_search)
    monkeypatch.setattr("app.agents.searcher._GATHER_TIMEOUT_SECONDS", 0.1)

    state = {"query": "timeout test", "plan": ["q1"]}
    with pytest.raises(SearchUnavailableError, match="timed out"):
        await searcher_run(state)
