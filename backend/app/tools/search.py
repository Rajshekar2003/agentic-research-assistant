"""Tavily search tool for the Agentic Research Assistant.

Tavily is chosen over alternatives (SerpAPI, Bing, DuckDuckGo scraping) because it is
purpose-built for LLM retrieval pipelines: it returns clean, deduplicated snippets rather
than raw HTML, filters low-quality or ad-heavy results, and exposes relevance scores so
callers can rank or filter before feeding context into a prompt.

The module exposes a single async `search()` function backed by a module-level singleton
client (via `get_search_client()` with @lru_cache). This mirrors the LLM client pattern
so lifetime and configuration loading are consistent across all external I/O.
"""

import asyncio
import logging
from functools import lru_cache

from pydantic import BaseModel, Field
from tavily import AsyncTavilyClient

from app.config import get_settings

logger = logging.getLogger(__name__)

_TIMEOUT_SECONDS = 15.0


class SearchUnavailableError(Exception):
    """Raised when Tavily fails for any reason (rate limit, network, quota, etc.).

    The message is always a sanitized string — raw Tavily error details are never
    surfaced so that API keys, quota levels, and internal error shapes stay private.
    """


class SearchResult(BaseModel):
    """A single result returned by the Tavily search API.

    Args:
        title: Display title of the source page.
        url: Canonical URL of the source (plain str for flexibility; validated
             as HttpUrl by the downstream Source schema).
        content: Excerpt or summary from the page (max 2000 chars).
        score: Tavily relevance score, or None if not returned by the API.
    """

    title: str = Field(..., min_length=1)
    url: str
    content: str = Field(..., max_length=2000)
    score: float | None = None


@lru_cache
def get_search_client() -> AsyncTavilyClient:
    """Return the singleton AsyncTavilyClient, constructed once from application settings.

    Returns:
        The cached AsyncTavilyClient instance.
    """
    return AsyncTavilyClient(api_key=get_settings().tavily_api_key.get_secret_value())


async def search(query: str, max_results: int = 5) -> list[SearchResult]:
    """Search the web via Tavily and return deduplicated, LLM-ready results.

    Args:
        query: Natural-language search query.
        max_results: Maximum number of results to return (applied after dedup).

    Returns:
        List of SearchResult objects, deduplicated by URL, up to `max_results` items.

    Raises:
        SearchUnavailableError: If Tavily fails for any reason (rate limit, network
            error, quota exhaustion, timeout). The original error type is logged but
            never propagated to callers.
    """
    client = get_search_client()
    try:
        response = await asyncio.wait_for(
            client.search(
                query=query,
                max_results=max_results,
                search_depth="basic",
                include_answer=False,
            ),
            timeout=_TIMEOUT_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Tavily search failed with %s", type(exc).__name__)
        raise SearchUnavailableError("Search service temporarily unavailable") from exc

    raw_results: list[dict] = response.get("results", [])

    seen_urls: set[str] = set()
    results: list[SearchResult] = []
    for item in raw_results:
        url = item.get("url", "")
        if url in seen_urls:
            continue
        seen_urls.add(url)
        results.append(
            SearchResult(
                title=item.get("title", "Untitled"),
                url=url,
                content=(item.get("content") or "")[:2000],
                score=item.get("score"),
            )
        )
        if len(results) >= max_results:
            break

    return results
