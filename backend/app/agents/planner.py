"""Planner agent for the multi-agent research graph.

Role: Decomposes the user query into 2-4 focused sub-questions that, taken together,
let a downstream Searcher fully address the original question.

Input fields:  state["query"] (str)
Output fields: state["plan"] (list[str]), plus LLM telemetry fields (provider, model,
               tokens_in, tokens_out) — these are overwritten by the Searcher's own
               LLM call (last-write-wins on telemetry fields).

Prompt strategy: Zero-shot decomposition with a strict JSON-only output constraint.
The LLM is instructed to return a bare JSON array with no preamble or markdown fencing.
On any parse or validation failure the agent falls back to [query] so downstream nodes
always receive a non-empty plan and the graph never crashes on bad LLM output.
"""

import json
import logging
import time

from app.graph.state import ResearchState
from app.llm.client import get_llm_client

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a research planner. Decompose the user's question into 2-4 specific, focused "
    "sub-questions that, taken together, would let a researcher answer the original question "
    "well. Output ONLY a JSON array of strings — no preamble, no explanation, no markdown "
    'code fence. Example: ["sub-question 1", "sub-question 2"]. If the original question is '
    "simple and atomic (e.g., a single factual lookup), output a single-element array "
    "containing the original question."
)


def _parse_plan(raw: str, fallback_query: str) -> list[str]:
    """Parse the LLM's JSON output into a validated plan list.

    Args:
        raw: Raw text returned by the LLM.
        fallback_query: The original query used as a single-element fallback on failure.

    Returns:
        A list of 1-4 non-empty, stripped sub-question strings.
    """
    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError:
        logger.warning("planner: JSON parse failure — raw output: %.200s", raw)
        return [fallback_query]

    if not isinstance(parsed, list):
        logger.warning("planner: expected list, got %s — falling back", type(parsed).__name__)
        return [fallback_query]

    items = [s.strip() for s in parsed if isinstance(s, str) and s.strip()]
    if not items:
        logger.warning("planner: empty or all-invalid items after filtering — falling back")
        return [fallback_query]

    return items[:4]


async def run(state: ResearchState) -> dict:
    """Decompose the query into a research plan (list of sub-questions).

    Calls the LLM with a JSON-only decomposition prompt and parses the result.
    Falls back to a single-element plan on any parse or validation failure so the
    graph is never blocked by a bad LLM response.

    Args:
        state: ResearchState containing at least 'query'.

    Returns:
        State delta with 'plan' (list[str]) and LLM telemetry fields (provider,
        model, tokens_in, tokens_out). The telemetry fields will be overwritten
        by the Searcher's own LLM call.

    Raises:
        LLMUnavailableError: If both Groq and Gemini fail. Propagates to the endpoint.
    """
    t0 = time.perf_counter()
    query = state["query"]
    llm = get_llm_client()

    result = await llm.complete(_SYSTEM_PROMPT, query, max_tokens=300, temperature=0.2)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    plan = _parse_plan(result.text, query)

    logger.info(
        "planner_node — plan_size=%d model=%s latency_ms=%d tokens_in=%s tokens_out=%s",
        len(plan),
        result.model,
        latency_ms,
        result.tokens_in,
        result.tokens_out,
    )

    return {
        "plan": plan,
        "provider": result.provider,
        "model": result.model,
        "tokens_in": result.tokens_in,
        "tokens_out": result.tokens_out,
    }
