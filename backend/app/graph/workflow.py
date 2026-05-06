"""LangGraph workflow for the Agentic Research Assistant.

Week 2 Day 9 — graph has 3 nodes: planner, searcher, fact_checker.
  START → planner → searcher → fact_checker → END

  Day 10 splits FactChecker's synthesis into a dedicated writer node (4 nodes).
  Day 11 adds critic with conditional edge back to writer (5 nodes).

The compiled graph is exposed via get_compiled_graph() (lru_cache singleton) and
as the module-level compiled_graph variable for direct import access.
"""

import logging
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.agents import planner, searcher
from app.agents.fact_checker import run as fact_checker_run
from app.graph.state import ResearchState

logger = logging.getLogger(__name__)


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
    builder.add_edge(START, "planner")
    builder.add_edge("planner", "searcher")
    builder.add_edge("searcher", "fact_checker")
    builder.add_edge("fact_checker", END)
    return builder.compile()


# Module-level singleton — importable directly as
# `from app.graph.workflow import compiled_graph`
compiled_graph = get_compiled_graph()
