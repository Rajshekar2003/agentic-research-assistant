"""Searcher agent for the multi-agent research graph.

Role: Runs Tavily search for each sub-question in the plan, deduplicates results
globally by URL, caps the combined result set at 8, and synthesizes a grounded
answer over the original query.

Input fields:  state["plan"] (list[str]), state["query"] (str)
Output fields: state["search_results"] (list[dict]), state["sources"] (list[Source]),
               state["final_answer"] (str), plus LLM telemetry fields (provider,
               model, tokens_in, tokens_out, elapsed_ms).

Prompt strategy: Identical to the Week 1 baseline endpoint — same system prompt,
same numbered context block format, same grounded-citation instructions. Keeping the
prompt byte-identical preserves eval validity: the Week 4 comparison between baseline
and multi-agent paths uses the same LLM generation quality signal.
"""

import logging
import time

from app.graph.state import ResearchState
from app.llm.client import get_llm_client
from app.schemas import Source
from app.tools.search import SearchResult, search

logger = logging.getLogger(__name__)

# Must remain identical to the system prompt in app/api/research.py.
# Any change confounds the Week 4 eval comparison.
_SYSTEM_PROMPT = (
    "You are a research assistant. Answer the user's question using ONLY the numbered sources "
    "provided below. Cite sources inline using [1], [2], etc. matching the source numbers. "
    "If the sources do not contain enough information to answer, say so explicitly. "
    "Do not invent facts. Do not cite sources that aren't in the provided list. "
    "Keep the answer concise — aim for 2-4 sentences unless the question genuinely needs more."
)


def _build_context(results: list[SearchResult]) -> str:
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


async def run(state: ResearchState) -> dict:
    """Search for evidence across all plan sub-questions and synthesize an answer.

    Calls Tavily once per sub-question, deduplicates the combined results by URL,
    caps the final set at 8 (highest-score first), then calls the LLM to synthesize
    a grounded answer over the original query using the collected evidence.

    Per-sub-question result count: max(2, 5 // len(plan)), keeping total context
    size bounded as the plan grows (4-element plan → 2 results each, 1-element
    plan → 5 results).

    Args:
        state: ResearchState containing 'plan' (list[str]) and 'query' (str).

    Returns:
        State delta with search_results (list[dict]), sources (list[Source]),
        final_answer, and LLM telemetry (provider, model, tokens_in, tokens_out,
        elapsed_ms for the LLM call only — Day 11 will sum across all nodes).

    Raises:
        SearchUnavailableError: If any Tavily call fails. Propagates to the endpoint.
        LLMUnavailableError: If both LLM providers fail. Propagates to the endpoint.
    """
    query = state["query"]
    plan = state["plan"]

    max_per_q = max(2, 5 // len(plan))

    all_results: list[SearchResult] = []
    for sub_question in plan:
        sub_results = await search(sub_question, max_results=max_per_q)
        all_results.extend(sub_results)

    # Global URL dedup — search() dedupes per-call; we need a second pass across calls.
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

    context = _build_context(deduped)
    user_prompt = f"{context}\n\nQuestion: {query}"

    llm = get_llm_client()
    result = await llm.complete(_SYSTEM_PROMPT, user_prompt)

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
        "final_results=%d llm_latency_ms=%d tokens_in=%s tokens_out=%s",
        len(plan),
        len(plan),
        unique_urls_count,
        len(deduped),
        result.latency_ms,
        result.tokens_in,
        result.tokens_out,
    )

    return {
        "search_results": [r.model_dump() for r in deduped],
        "sources": sources,
        "final_answer": result.text,
        "provider": result.provider,
        "model": result.model,
        "tokens_in": result.tokens_in,
        "tokens_out": result.tokens_out,
        "elapsed_ms": result.latency_ms,
    }
