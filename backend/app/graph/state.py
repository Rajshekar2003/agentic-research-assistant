from typing import TypedDict


class ResearchState(TypedDict):
    query: str           # input
    plan: list[str]      # Planner
    search_results: list  # Searcher
    facts: list[str]     # Fact-checker
    draft: str           # Writer
    critique: str        # Critic
    revision_count: int  # Critic loop guard
    final_answer: str    # Writer (after Critic approves)
    sources: list        # Searcher
