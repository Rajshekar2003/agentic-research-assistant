"""Fact-checker agent for the multi-agent research graph.

As of Day 9, FactChecker handles BOTH verification AND synthesis so the graph stays
runnable end-to-end. Day 8 had Searcher synthesize the final answer; that responsibility
moved here. Day 10 will split synthesis into a dedicated Writer node; FactChecker will
then handle verification only.

Input fields:  state["query"] (str), state["search_results"] (list[dict])
Output fields: state["facts"] (list[dict]), state["final_answer"] (str),
               plus LLM telemetry fields (provider, model, tokens_in, tokens_out).

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
    "sources, you do two things in order:\n\n"
    "STEP 1 — Extract verified facts:\n"
    "List ONLY claims that are explicitly supported by the sources. For each fact, include the "
    "source numbers that support it. Do not infer, generalize, or add background knowledge. If "
    "the sources contradict each other, note both claims separately. If the sources do not "
    "contain enough information to answer the question, say so explicitly in the answer rather "
    "than inventing facts.\n\n"
    "STEP 2 — Write the answer:\n"
    "Use ONLY the facts from STEP 1 to answer the user's question. Cite source numbers inline "
    "as [1], [2], etc. Keep it concise (2-5 sentences unless the question genuinely needs more). "
    "Do not introduce claims that weren't in STEP 1.\n\n"
    "Output format — return a single JSON object with this exact shape, and nothing else (no "
    "markdown fences, no preamble):\n\n"
    "{\n"
    '  "facts": [\n'
    '    {"claim": "string describing the verified fact", "sources": [1, 3]},\n'
    '    {"claim": "...", "sources": [2]}\n'
    "  ],\n"
    '  "answer": "the final synthesized answer with [1], [2] inline citations"\n'
    "}"
)

_FALLBACK_ANSWER = "I couldn't verify any claims from the available sources for this question."


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


def _parse_response(raw: str, max_source_idx: int) -> tuple[list[dict], str]:
    """Parse the LLM's JSON output into (facts, answer).

    Returns ([], FALLBACK_ANSWER) on parse failure or schema mismatch, logging a
    WARNING with the raw response so bad outputs are diagnosable without crashing.
    """
    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError:
        logger.warning("fact_checker: JSON parse failure — raw: %.300s", raw)
        return [], _FALLBACK_ANSWER

    if not isinstance(parsed, dict) or "facts" not in parsed or "answer" not in parsed:
        logger.warning("fact_checker: schema mismatch — raw: %.300s", raw)
        return [], _FALLBACK_ANSWER

    if not isinstance(parsed["facts"], list) or not isinstance(parsed["answer"], str):
        logger.warning("fact_checker: unexpected types in parsed object — raw: %.300s", raw)
        return [], _FALLBACK_ANSWER

    facts = _validate_and_sanitize_facts(parsed["facts"], max_source_idx)
    answer = parsed["answer"].strip() or _FALLBACK_ANSWER
    return facts, answer


async def run(state: ResearchState) -> dict:
    """Extract verified facts and synthesize a grounded answer from search results.

    Calls the LLM with a strict JSON-output fact-checking prompt. On any parse or
    schema failure, falls back to empty facts and an honest "couldn't verify" answer
    rather than fabricating claims.

    Args:
        state: ResearchState with 'query' (str) and 'search_results' (list[dict]).

    Returns:
        State delta with 'facts' (list[dict]), 'final_answer' (str), and LLM telemetry
        fields (provider, model, tokens_in, tokens_out).

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

    facts, final_answer = _parse_response(result.text, max_source_idx=len(search_results))

    logger.info(
        "fact_checker_node — facts_count=%d answer_length=%d model=%s "
        "latency_ms=%d tokens_in=%s tokens_out=%s",
        len(facts),
        len(final_answer),
        result.model,
        latency_ms,
        result.tokens_in,
        result.tokens_out,
    )

    return {
        "facts": facts,
        "final_answer": final_answer,
        "provider": result.provider,
        "model": result.model,
        "tokens_in": result.tokens_in,
        "tokens_out": result.tokens_out,
    }
