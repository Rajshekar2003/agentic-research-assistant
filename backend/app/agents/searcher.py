"""Searcher agent for the multi-agent research graph.

Day 9: retrieval-only node; LLM synthesis moved to FactChecker.
Day 12: per-sub-question Tavily calls now run in parallel via asyncio.gather,
        reducing latency from ~N*2s (sequential) to ~2s regardless of plan size.
        A 20s overall gather timeout bounds worst-case fan-out.

Role: Runs Tavily search for each sub-question in the plan, deduplicates results
globally by URL, and caps the combined result set at 8.

Input fields:  state["plan"] (list[str]), state["query"] (str)
Output fields: state["search_results"] (list[dict]), state["sources"] (list[Source]).
               LLM telemetry fields are explicitly set to None so FactChecker (and
               later Writer) overwrite them with their own values.
"""

import asyncio
import logging
import time

from app.graph.state import ResearchState
from app.schemas import Source
from app.tools.search import SearchResult, SearchUnavailableError, search

logger = logging.getLogger(__name__)

_GATHER_TIMEOUT_SECONDS = 20.0


async def run(state: ResearchState) -> dict:
    """Search for evidence across all plan sub-questions in parallel.

    Launches one Tavily call per sub-question concurrently via asyncio.gather,
    deduplicates the combined results by URL, and caps the final set at 8
    (highest-score first). Synthesis is handled by FactChecker as of Day 9.

    Per-sub-question result count: max(2, 5 // len(plan)), keeping total context
    size bounded as the plan grows (4-element plan → 2 results each, 1-element
    plan → 5 results).

    Args:
        state: ResearchState containing 'plan' (list[str]) and 'query' (str).

    Returns:
        State delta with search_results (list[dict]), sources (list[Source]),
        and LLM telemetry fields explicitly set to None (FactChecker overwrites).

    Raises:
        SearchUnavailableError: If any Tavily call fails or the overall gather
            exceeds _GATHER_TIMEOUT_SECONDS. Propagates to the endpoint.
    """
    plan = state["plan"]
    max_per_q = max(2, 5 // len(plan))

    t0 = time.perf_counter()
    try:
        results_per_q: list[list[SearchResult]] = await asyncio.wait_for(
            asyncio.gather(*[search(sub_q, max_results=max_per_q) for sub_q in plan]),
            timeout=_GATHER_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError as exc:
        raise SearchUnavailableError("Search timed out across sub-questions") from exc
    elapsed_ms = int((time.perf_counter() - t0) * 1000)

    all_results: list[SearchResult] = []
    for sub_results in results_per_q:
        all_results.extend(sub_results)

    # Global URL dedup — search() dedupes per-call; second pass across parallel calls.
    seen_urls: set[str] = set()
    deduped: list[SearchResult] = []
    for r in all_results:
        if r.url not in seen_urls:
            seen_urls.add(r.url)
            deduped.append(r)

    unique_urls_count = len(deduped)

    # Cap at 8 results; prefer highest score, fall back to insertion order for unscored.
    if unique_urls_count > 8:
        scored = sorted(
            [r for r in deduped if r.score is not None],
            key=lambda r: r.score,  # type: ignore[arg-type]
            reverse=True,
        )
        unscored = [r for r in deduped if r.score is None]
        deduped = (scored + unscored)[:8]

    sources = [
        Source(
            title=r.title,
            url=r.url.strip(),
            snippet=r.content[:2000],
        )
        for r in deduped
    ]

    logger.info(
        "searcher_node — sub_questions=%d total_search_calls=%d unique_urls=%d "
        "final_results=%d parallel=true elapsed_ms=%d",
        len(plan),
        len(plan),
        unique_urls_count,
        len(deduped),
        elapsed_ms,
    )

    return {
        "search_results": [r.model_dump() for r in deduped],
        "sources": sources,
        "provider": None,
        "model": None,
        "tokens_in": None,
        "tokens_out": None,
    }
