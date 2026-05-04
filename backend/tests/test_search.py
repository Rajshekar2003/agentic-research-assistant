"""Tests for the Tavily search wrapper (app.tools.search)."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.tools.search import SearchResult, SearchUnavailableError, search


def _make_tavily_response(items: list[dict]) -> dict:
    """Build a fake Tavily API response dict matching the real SDK shape."""
    return {"results": items}


def _make_item(
    title: str, url: str, content: str, score: float | None = 0.9
) -> dict:
    return {"title": title, "url": url, "content": content, "score": score}


@pytest.fixture(autouse=True)
def clear_search_cache():
    """Clear the search client lru_cache before and after each test."""
    from app.tools.search import get_search_client

    get_search_client.cache_clear()
    yield
    get_search_client.cache_clear()


async def test_search_returns_results():
    """Mock AsyncTavilyClient.search; assert SearchResult fields populate correctly
    and duplicate URLs are deduplicated."""
    fake_items = [
        _make_item("Title One", "https://example.com/one", "Content one", 0.95),
        _make_item("Title Two", "https://example.com/two", "Content two", 0.80),
        # Duplicate URL — should be dropped by dedup logic
        _make_item("Title One Dup", "https://example.com/one", "Duplicate content", 0.70),
    ]
    mock_client = MagicMock()
    mock_client.search = AsyncMock(return_value=_make_tavily_response(fake_items))

    with patch("app.tools.search.get_search_client", return_value=mock_client):
        results = await search("test query", max_results=5)

    assert len(results) == 2
    assert isinstance(results[0], SearchResult)
    assert results[0].title == "Title One"
    assert results[0].url == "https://example.com/one"
    assert results[0].content == "Content one"
    assert results[0].score == 0.95
    assert results[1].url == "https://example.com/two"


async def test_search_raises_search_unavailable_on_failure():
    """Any exception from the Tavily client raises SearchUnavailableError
    with a sanitized message — original error class name must not appear."""
    mock_client = MagicMock()
    mock_client.search = AsyncMock(side_effect=RuntimeError("Tavily internal quota exceeded"))

    with patch("app.tools.search.get_search_client", return_value=mock_client):
        with pytest.raises(SearchUnavailableError) as exc_info:
            await search("test query")

    assert "RuntimeError" not in str(exc_info.value)
    assert str(exc_info.value) == "Search service temporarily unavailable"


async def test_search_respects_max_results():
    """Returns at most max_results items even when mock returns more."""
    fake_items = [
        _make_item(f"Title {i}", f"https://example.com/{i}", f"Content {i}")
        for i in range(10)
    ]
    mock_client = MagicMock()
    mock_client.search = AsyncMock(return_value=_make_tavily_response(fake_items))

    with patch("app.tools.search.get_search_client", return_value=mock_client):
        results = await search("test query", max_results=3)

    assert len(results) == 3
