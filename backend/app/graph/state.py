"""Shared state passed between LangGraph nodes.

Day 9 — `facts` is populated by FactChecker as a list of dicts, each with a
verified claim and the source IDs that support it (e.g. {"claim": "...", "sources": [1, 3]}).
Currently exposed in API responses; frontend rendering is a Day 13 stretch goal.

Day 11 — `critic_verdict`, `critique`, and `revision_count` wire the Critic → Writer
feedback loop. `revision_count` is incremented ONLY by Critic on each "revise" verdict.
`critic_verdict` is consumed by the conditional edge in workflow.py.

In Week 1 only the searcher and writer nodes touch this state; Week 2 adds
planner/fact_checker/critic. Fields are populated incrementally — early-week
nodes leave later-week fields empty.
"""

from typing import Literal, TypedDict


class ResearchState(TypedDict, total=False):
    query: str            # input — caller (Week 1)
    plan: list[str]       # Planner — Week 2
    search_results: list  # Searcher — Week 1 Day 6
    facts: list[dict]     # FactChecker — each dict: {"claim": str, "sources": list[int]}
    draft: str            # Writer — Week 2
    critique: str         # Critic feedback message; empty string when approved (Day 11)
    revision_count: int   # incremented by Critic on each "revise" verdict; default 0 (Day 11)
    critic_verdict: Literal["approve", "revise"] | None  # set by Critic; None before Critic runs
    final_answer: str     # Writer (post-Critic approval) — Week 1 Day 6 (populated directly)
    sources: list         # Searcher — Week 1 Day 6
    elapsed_ms: int       # graph execution time, set by graph wrapper
    provider: str         # LLM provider that served the writer node, for telemetry
    model: str            # LLM model name that served the writer node
    tokens_in: int | None   # LLM input token count — Week 1 Day 6
    tokens_out: int | None  # LLM output token count — Week 1 Day 6
