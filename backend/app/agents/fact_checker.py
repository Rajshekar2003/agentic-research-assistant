"""Fact-checker agent for the multi-agent research graph.

Verification-only as of Day 10. Earlier, this node also synthesized the answer;
that responsibility moved to Writer.

Input fields:  state["query"] (str), state["search_results"] (list[dict])
Output fields: state["facts"] (list[dict]), plus LLM telemetry fields
               (provider, model, tokens_in, tokens_out).

Each fact has shape: {"claim": str, "sources": list[int]}.
Source IDs are 1-indexed and sanitized against the actual search_results length.
"""

import json
import logging
import time

from app.graph.state import ResearchState
from app.llm.client import get_llm_client

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a fact-checking research assistant. Given a user's question and a numbered list of "
    "sources, extract claims from the sources that are relevant to the question.\n\n"
    "For each fact, include the source numbers that explicitly support it. Do not infer beyond "
    "what the sources state. Do not add background knowledge. If sources contradict each other, "
    "list both claims separately. If the sources do not contain information relevant to the "
    "question, return an empty facts list.\n\n"
    "Output format — return a single JSON object with this exact shape, and nothing else (no "
    "markdown fences, no preamble):\n\n"
    "{\n"
    '  "facts": [\n'
    '    {"claim": "string describing the verified fact", "sources": [1, 3]},\n'
    '    {"claim": "...", "sources": [2]}\n'
    "  ]\n"
    "}"
)


def _build_context(search_results: list[dict]) -> str:
    """Format search results as a numbered context block."""
    lines = []
    for i, r in enumerate(search_results, start=1):
        lines.append(f"[{i}] {r.get('title', '')}")
        lines.append(f"URL: {r.get('url', '')}")
        lines.append(r.get("content", ""))
        lines.append("")
    return "\n".join(lines).rstrip()


def _validate_and_sanitize_facts(raw_facts: list, max_source_idx: int) -> list[dict]:
    """Validate fact dicts and drop malformed entries or out-of-range source IDs.

    Source IDs outside [1, max_source_idx] are filtered out. If that leaves a fact
    with no valid sources, the fact itself is dropped rather than kept unsupported.
    """
    valid: list[dict] = []
    for item in raw_facts:
        if not isinstance(item, dict):
            continue
        claim = item.get("claim")
        sources = item.get("sources")
        if not isinstance(claim, str) or not claim.strip():
            continue
        if not isinstance(sources, list):
            continue
        sanitized = [s for s in sources if isinstance(s, int) and 1 <= s <= max_source_idx]
        if not sanitized:
            continue
        valid.append({"claim": claim.strip(), "sources": sanitized})
    return valid


def _parse_response(raw: str, max_source_idx: int) -> list[dict]:
    """Parse the LLM's JSON output into a validated list of fact dicts.

    Returns [] on parse failure or schema mismatch, logging a WARNING with the raw
    response so bad outputs are diagnosable without crashing.
    """
    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError:
        logger.warning("fact_checker: JSON parse failure — raw: %.300s", raw)
        return []

    if not isinstance(parsed, dict) or "facts" not in parsed:
        logger.warning("fact_checker: schema mismatch — raw: %.300s", raw)
        return []

    if not isinstance(parsed["facts"], list):
        logger.warning("fact_checker: unexpected types in parsed object — raw: %.300s", raw)
        return []

    return _validate_and_sanitize_facts(parsed["facts"], max_source_idx)


async def run(state: ResearchState) -> dict:
    """Extract verified facts from search results.

    Calls the LLM with a strict JSON-output fact-checking prompt. On any parse or
    schema failure, falls back to an empty facts list rather than fabricating claims.
    Synthesis of the final answer is handled by the Writer node.

    Args:
        state: ResearchState with 'query' (str) and 'search_results' (list[dict]).

    Returns:
        State delta with 'facts' (list[dict]) and LLM telemetry fields
        (provider, model, tokens_in, tokens_out). Does not set 'final_answer'.

    Raises:
        LLMUnavailableError: If both Groq and Gemini fail. Propagates to the endpoint.
    """
    t0 = time.perf_counter()
    query = state["query"]
    search_results = state.get("search_results") or []

    context = _build_context(search_results)
    user_prompt = f"{context}\n\nQuestion: {query}"

    llm = get_llm_client()
    result = await llm.complete(_SYSTEM_PROMPT, user_prompt, max_tokens=1200, temperature=0.2)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    facts = _parse_response(result.text, max_source_idx=len(search_results))

    logger.info(
        "fact_checker_node — facts_count=%d model=%s latency_ms=%d tokens_in=%s tokens_out=%s",
        len(facts),
        result.model,
        latency_ms,
        result.tokens_in,
        result.tokens_out,
    )

    return {
        "facts": facts,
        "provider": result.provider,
        "model": result.model,
        "tokens_in": result.tokens_in,
        "tokens_out": result.tokens_out,
    }
