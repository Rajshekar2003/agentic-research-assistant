"""LangGraph workflow for the Agentic Research Assistant.

Week 2 Day 11 — graph has 5 nodes with a Critic→Writer feedback loop capped at 2 revisions.

  START → planner → searcher → fact_checker → writer → critic
                                                  ↑           ↓
                                                  └─ revise ──┤ (revision_count < 2)
                                                              ↓
                                                             END (approve OR revision_count >= 2)

revision_count is incremented by Critic on each "revise" verdict and checked by
route_after_critic.  Hard cap prevents infinite loops if Critic stays unsatisfied.

The compiled graph is exposed via get_compiled_graph() (lru_cache singleton) and
as the module-level compiled_graph variable for direct import access.
"""

import logging
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.agents import planner, searcher
from app.agents.critic import run as critic_run
from app.agents.fact_checker import run as fact_checker_run
from app.agents.writer import run as writer_run
from app.graph.state import ResearchState

logger = logging.getLogger(__name__)


def route_after_critic(state: ResearchState) -> str:
    """Route the graph after Critic evaluates the Writer's draft.

    Routing rules:
    - verdict == "approve" → END (draft accepted, ship it)
    - verdict == "revise" AND revision_count < 2 → "writer" (one more revision allowed)
    - verdict == "revise" AND revision_count >= 2 → END (hard cap hit, ship last draft)

    revision_count is incremented by the Critic node BEFORE this function fires, so
    a value of 1 means one revision has been requested (and the current Writer run is
    about to be the first revision); a value of 2 means the cap is reached.

    Args:
        state: The merged ResearchState after the critic node has run.

    Returns:
        "writer" to loop back for a revision, or END to terminate the graph.
    """
    verdict = state.get("critic_verdict")
    revision_count = state.get("revision_count", 0)
    if verdict == "revise" and revision_count < 2:
        return "writer"
    return END


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
    builder.add_node("planner", planner.run)
    builder.add_node("searcher", searcher.run)
    builder.add_node("fact_checker", fact_checker_run)
    builder.add_node("writer", writer_run)
    builder.add_node("critic", critic_run)
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "searcher")
    builder.add_edge("searcher", "fact_checker")
    builder.add_edge("fact_checker", "writer")
    builder.add_edge("writer", "critic")
    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {"writer": "writer", END: END},
    )
    return builder.compile()


# Module-level singleton — importable directly as
# `from app.graph.workflow import compiled_graph`
compiled_graph = get_compiled_graph()
