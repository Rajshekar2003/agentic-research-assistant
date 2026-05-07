"""Research endpoint — search-augmented generation (RAG) baseline.

# Day 4 done: Real LLM client (Groq primary, Gemini fallback).
# Day 5 done: Tavily search → context-formatted prompt → grounded answer with [1]/[2] citations.
# Day 6 done: POST /research/graph wraps the same pipeline in a single-node LangGraph graph.
# Day 8 done: 2-node graph (planner → searcher).
# Day 9 done: 3-node graph (planner → searcher → fact_checker); ResearchResponse exposes facts.
# Day 10 done: 4-node graph; fact_checker is verification-only; writer synthesizes final answer.
# Day 11 done: 5-node graph; Critic with conditional feedback loop back to Writer (capped at 2).
# TODO Days 12-13: stress test, parallelize Searcher's Tavily calls, tune prompts.
"""

import logging
import time

from fastapi import APIRouter, HTTPException

from app.graph.workflow import get_compiled_graph
from app.llm.client import LLMUnavailableError, get_llm_client
from app.schemas import Fact, ResearchRequest, ResearchResponse, Source
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


@router.post("/graph", response_model=ResearchResponse)
async def run_research_graph(request: ResearchRequest) -> ResearchResponse:
    """Run the query through the multi-agent LangGraph state machine.

    Invokes the compiled 5-node graph (Planner → Searcher → FactChecker → Writer → Critic)
    with the query and maps the resulting state into a ResearchResponse.  The Critic may
    route back to Writer up to 2 times; the revision loop is transparent to callers except
    for the critic_verdict and revisions fields in the response.  LLM telemetry in the log
    line reflects the last Writer node call (state fields are last-write-wins).

    # Day 11 done: 5-node graph; Critic with conditional feedback loop capped at 2 revisions.
    # TODO Week 4: eval harness will hit this endpoint for graph-mode runs.

    Args:
        request: Validated research request containing the query string.

    Returns:
        ResearchResponse with grounded answer, source list, verified facts,
        critic verdict, revision count, wall-clock elapsed time, and mode="graph".

    Raises:
        HTTPException: 503 if search is unavailable or both LLM providers fail.
    """
    start = time.perf_counter()
    logger.info("Research graph query received: %.100s path=graph", request.query)

    try:
        state = await get_compiled_graph().ainvoke({"query": request.query})
    except SearchUnavailableError:
        raise HTTPException(status_code=503, detail="Search service temporarily unavailable")
    except LLMUnavailableError:
        raise HTTPException(status_code=503, detail="Research service temporarily unavailable")

    elapsed_ms = int((time.perf_counter() - start) * 1000)

    facts_raw = state.get("facts")
    facts = [Fact(**f) for f in facts_raw] if facts_raw is not None else None
    critic_verdict = state.get("critic_verdict")
    revisions = state.get("revision_count", 0)

    logger.info(
        "Research graph complete: path=graph provider=%s model=%s "
        "search_results_count=%d elapsed_ms=%d tokens_in=%s tokens_out=%s "
        "plan_size=%d facts_count=%d critic_verdict=%s revisions=%d",
        state.get("provider"),
        state.get("model"),
        len(state.get("search_results") or []),
        elapsed_ms,
        state.get("tokens_in"),
        state.get("tokens_out"),
        len(state.get("plan") or []),
        len(facts_raw or []),
        critic_verdict,
        revisions,
    )

    return ResearchResponse(
        answer=state["final_answer"],
        sources=state["sources"],
        elapsed_ms=elapsed_ms,
        mode="graph",
        facts=facts,
        critic_verdict=critic_verdict,
        revisions=revisions,
    )
