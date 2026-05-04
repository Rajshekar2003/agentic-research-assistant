"""Research endpoint — search-augmented generation (RAG) baseline.

# Day 4 done: Real LLM client (Groq primary, Gemini fallback).
# Day 5 done: Tavily search → context-formatted prompt → grounded answer with [1]/[2] citations.
# Day 6 TODO: Wrap in LangGraph state machine for multi-agent evaluation.
"""

import logging
import time

from fastapi import APIRouter, HTTPException

from app.llm.client import LLMUnavailableError, get_llm_client
from app.schemas import ResearchRequest, ResearchResponse, Source
from app.tools.search import SearchUnavailableError, search

router = APIRouter(prefix="/research")
logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a research assistant. Answer the user's question using ONLY the numbered sources "
    "provided below. Cite sources inline using [1], [2], etc. matching the source numbers. "
    "If the sources do not contain enough information to answer, say so explicitly. "
    "Do not invent facts. Do not cite sources that aren't in the provided list. "
    "Keep the answer concise — aim for 2-4 sentences unless the question genuinely needs more."
)


def _build_context(results: list) -> str:
    """Format search results as a numbered context block for the LLM prompt.

    Args:
        results: List of SearchResult objects.

    Returns:
        A multi-line string with each source numbered [1], [2], etc.
    """
    lines = []
    for i, r in enumerate(results, start=1):
        lines.append(f"[{i}] {r.title}")
        lines.append(f"URL: {r.url}")
        lines.append(r.content)
        lines.append("")
    return "\n".join(lines).rstrip()


@router.post("", response_model=ResearchResponse)
async def run_research(request: ResearchRequest) -> ResearchResponse:
    """Run the RAG baseline: search → ground prompt → generate grounded answer.

    # Day 4 done: Real LLM client (Groq primary, Gemini fallback).
    # Day 5 done: Tavily search → context-formatted prompt → grounded answer.
    # Day 6 TODO: LangGraph wrapping for multi-agent eval comparison.

    Args:
        request: Validated research request containing the query string.

    Returns:
        ResearchResponse with grounded answer, source list, wall-clock elapsed time,
        and mode set to "baseline".

    Raises:
        HTTPException: 503 if search is unavailable or both LLM providers fail.
    """
    start = time.perf_counter()
    logger.info("Research query received: %.100s", request.query)

    # --- Search (must succeed — 503 on failure preserves baseline grounding) ---
    search_start = time.perf_counter()
    try:
        results = await search(request.query, max_results=5)
    except SearchUnavailableError:
        raise HTTPException(status_code=503, detail="Search service temporarily unavailable")
    search_latency_ms = int((time.perf_counter() - search_start) * 1000)

    # --- Build grounded prompt ---
    context = _build_context(results)
    user_prompt = f"{context}\n\nQuestion: {request.query}"

    # --- LLM generation ---
    llm = get_llm_client()
    try:
        result = await llm.complete(_SYSTEM_PROMPT, user_prompt)
    except LLMUnavailableError:
        raise HTTPException(status_code=503, detail="Research service temporarily unavailable")

    total_elapsed_ms = int((time.perf_counter() - start) * 1000)

    logger.info(
        "Research complete: provider=%s model=%s search_results_count=%d "
        "search_latency_ms=%d latency_ms=%s tokens_in=%s tokens_out=%s "
        "total_elapsed_ms=%d",
        result.provider,
        result.model,
        len(results),
        search_latency_ms,
        result.latency_ms,
        result.tokens_in,
        result.tokens_out,
        total_elapsed_ms,
    )

    sources = [
        Source(
            title=r.title,
            url=r.url.strip(),
            snippet=r.content[:2000],
        )
        for r in results
    ]

    return ResearchResponse(
        answer=result.text,
        sources=sources,
        elapsed_ms=total_elapsed_ms,
        mode="baseline",
    )
