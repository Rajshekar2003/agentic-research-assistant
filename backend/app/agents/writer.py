"""Writer agent for the multi-agent research graph.

Synthesis responsibility moved here from FactChecker as of Day 10. FactChecker now
only extracts verified {claim, sources} pairs; Writer takes those pairs and produces
the final grounded answer.

When facts is empty, Writer skips the LLM call entirely and returns a graceful
fallback message — preserving the honest-fallback behaviour FactChecker used to
provide on queries where no claims could be verified.

Input fields:  state["query"] (str), state["facts"] (list[dict])
Output fields: state["final_answer"] (str), plus LLM telemetry fields
               (provider, model, tokens_in, tokens_out). When facts are empty all
               telemetry fields are None (LLM was not called).
"""

import logging
import time

from app.graph.state import ResearchState
from app.llm.client import get_llm_client

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a research writer. Given a list of verified facts and the user's original "
    "question, synthesize a concise answer.\n\n"
    "Rules:\n"
    "- Use ONLY the verified facts. Do not introduce new information, background knowledge, "
    "or speculation.\n"
    "- Cite source numbers inline as [1], [2], etc. matching the source IDs from each fact.\n"
    "- Keep the answer concise (2-5 sentences unless the question genuinely needs more).\n"
    "- If a fact has multiple supporting sources, cite all relevant ones (e.g., [1, 3]).\n"
    '- Do not start the answer with phrases like "Based on the provided facts..." — just '
    "write the answer directly.\n"
    "- Output the answer text only. No JSON, no markdown headers, no preamble."
)

_FALLBACK_ANSWER = "I couldn't verify any claims from the available sources for this question."


def _build_facts_context(facts: list[dict]) -> str:
    """Format verified facts as a numbered context block for the Writer prompt.

    Args:
        facts: List of fact dicts, each with 'claim' (str) and 'sources' (list[int]).

    Returns:
        A multi-line string listing each fact with its supporting source IDs.
    """
    lines = ["Verified facts:"]
    for i, fact in enumerate(facts, start=1):
        src_ids = ", ".join(str(s) for s in fact.get("sources", []))
        lines.append(f"{i}. {fact['claim']} (sources: [{src_ids}])")
    return "\n".join(lines)


async def run(state: ResearchState) -> dict:
    """Synthesize a grounded final answer from the verified facts.

    If facts is empty, skips the LLM call and returns an honest fallback message.
    Otherwise, builds a context block from the verified facts and calls the LLM to
    produce a concise, citation-bearing answer.

    Args:
        state: ResearchState with 'query' (str) and 'facts' (list[dict]).

    Returns:
        State delta with 'final_answer' (str) and LLM telemetry fields
        (provider, model, tokens_in, tokens_out). Telemetry fields are all None
        when the LLM call is skipped due to empty facts.

    Raises:
        LLMUnavailableError: If both Groq and Gemini fail. Propagates to the endpoint.
    """
    facts = state.get("facts") or []
    query = state["query"]

    if not facts:
        logger.info("writer_node — skipped=true reason=no_facts")
        return {
            "final_answer": _FALLBACK_ANSWER,
            "provider": None,
            "model": None,
            "tokens_in": None,
            "tokens_out": None,
        }

    t0 = time.perf_counter()
    facts_context = _build_facts_context(facts)
    user_prompt = f"{facts_context}\n\nQuestion: {query}\n\nAnswer:"

    llm = get_llm_client()
    result = await llm.complete(_SYSTEM_PROMPT, user_prompt, max_tokens=600, temperature=0.3)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    final_answer = result.text.strip()

    logger.info(
        "writer_node — answer_length=%d model=%s latency_ms=%d tokens_in=%s tokens_out=%s",
        len(final_answer),
        result.model,
        latency_ms,
        result.tokens_in,
        result.tokens_out,
    )

    return {
        "final_answer": final_answer,
        "provider": result.provider,
        "model": result.model,
        "tokens_in": result.tokens_in,
        "tokens_out": result.tokens_out,
    }
