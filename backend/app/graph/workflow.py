"""LangGraph workflow for the Agentic Research Assistant.

This is the Week 1 single-node baseline graph; Week 2 will replace the single
research_node with a planner → searcher → fact_checker → writer → critic flow.
The compiled graph is exposed via get_compiled_graph() (lru_cache singleton) and
as the module-level compiled_graph variable for direct import access.
"""

import logging
import time
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.graph.state import ResearchState
from app.llm.client import get_llm_client
from app.schemas import Source
from app.tools.search import search

logger = logging.getLogger(__name__)

# Prompt constants are duplicated from app.api.research rather than imported to
# avoid coupling the graph module to the API layer.  A cross-import would risk
# circular dependencies in Week 2 when nodes are split into submodules that may
# themselves import from the API package.
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


async def research_node(state: ResearchState) -> dict:
    """Single research node: full baseline pipeline (search + generate).

    Reads query from state, calls Tavily search, builds a grounded prompt, and
    calls the LLM client.  Returns only the state delta; LangGraph merges it
    into the running state.

    Args:
        state: Current ResearchState containing at least 'query'.

    Returns:
        State delta dict with search_results, sources, final_answer, provider,
        model, tokens_in, tokens_out, and elapsed_ms populated.
    """
    node_start = time.perf_counter()
    query = state["query"]

    results = await search(query, max_results=5)

    context = _build_context(results)
    user_prompt = f"{context}\n\nQuestion: {query}"

    llm = get_llm_client()
    result = await llm.complete(_SYSTEM_PROMPT, user_prompt)

    elapsed_ms = int((time.perf_counter() - node_start) * 1000)

    sources = [
        Source(
            title=r.title,
            url=r.url.strip(),
            snippet=r.content[:2000],
        )
        for r in results
    ]

    logger.info(
        "research_node complete: provider=%s model=%s elapsed_ms=%d tokens_in=%s tokens_out=%s",
        result.provider,
        result.model,
        elapsed_ms,
        result.tokens_in,
        result.tokens_out,
    )

    return {
        "search_results": results,
        "sources": sources,
        "final_answer": result.text,
        "provider": result.provider,
        "model": result.model,
        "tokens_in": result.tokens_in,
        "tokens_out": result.tokens_out,
        "elapsed_ms": elapsed_ms,
    }


@lru_cache
def get_compiled_graph():
    """Return the singleton compiled LangGraph graph.

    The graph is compiled once and cached via lru_cache.  Compilation is
    idempotent but explicit caching documents the singleton intent and avoids
    repeated StateGraph construction on hot paths.

    Returns:
        The compiled Pregel graph instance (langgraph.pregel.Pregel).
    """
    builder = StateGraph(ResearchState)
    builder.add_node("research_node", research_node)
    builder.add_edge(START, "research_node")
    builder.add_edge("research_node", END)
    return builder.compile()


# Module-level singleton — importable directly as
# `from app.graph.workflow import compiled_graph`
compiled_graph = get_compiled_graph()
