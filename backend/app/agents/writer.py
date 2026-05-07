"""Writer agent for the multi-agent research graph.

Synthesis responsibility moved here from FactChecker as of Day 10. FactChecker now
only extracts verified {claim, sources} pairs; Writer takes those pairs and produces
the final grounded answer.

Day 11 — Writer is now revision-aware.  On a revision pass (revision_count > 0) it
receives the Critic's concrete feedback via state["critique"] and state["final_answer"]
(the previous draft), and produces a targeted revision using a different system prompt.
The first-pass prompt and logic are UNCHANGED from Day 10.

When facts is empty, Writer skips the LLM call entirely and returns a graceful
fallback message — regardless of revision_count, because there are no facts to
ground a revision against.

Input fields:  state["query"] (str), state["facts"] (list[dict]),
               state.get("revision_count", 0) (int),
               state.get("critique", "") (str — Critic's feedback on revision passes),
               state.get("final_answer", "") (str — previous draft on revision passes)
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

_REVISION_SYSTEM_PROMPT = (
    "You are a research writer revising an earlier draft. The earlier draft was reviewed by "
    "a critic who identified specific problems. Produce a revised draft that addresses every "
    "concrete issue in the critique while keeping the parts the critic did not object to.\n\n"
    "Rules (unchanged from first pass):\n"
    "- Use ONLY the verified facts. Do not introduce new information.\n"
    "- Cite sources inline as [1], [2], etc.\n"
    "- Keep concise (2-5 sentences usually).\n"
    "- Output the revised answer text only. No preamble, no JSON."
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
    """Synthesize or revise a grounded final answer from the verified facts.

    On the first pass (revision_count == 0) uses the standard synthesis prompt.
    On revision passes (revision_count > 0) uses the revision prompt, incorporating
    the Critic's concrete feedback and the previous draft.

    If facts is empty, skips the LLM call and returns an honest fallback message
    regardless of revision_count — there are no facts to ground a revision against.

    Args:
        state: ResearchState with 'query' (str), 'facts' (list[dict]), and optionally
               'revision_count' (int), 'critique' (str), 'final_answer' (str).

    Returns:
        State delta with 'final_answer' (str) and LLM telemetry fields
        (provider, model, tokens_in, tokens_out). Telemetry fields are all None
        when the LLM call is skipped due to empty facts.

    Raises:
        LLMUnavailableError: If both Groq and Gemini fail. Propagates to the endpoint.
    """
    facts = state.get("facts") or []
    query = state["query"]
    revision_count = state.get("revision_count", 0)
    is_revision = revision_count > 0

    if not facts:
        logger.info("writer_node — skipped=true reason=no_facts revision=%s", is_revision)
        return {
            "final_answer": _FALLBACK_ANSWER,
            "provider": None,
            "model": None,
            "tokens_in": None,
            "tokens_out": None,
        }

    t0 = time.perf_counter()
    facts_context = _build_facts_context(facts)

    if is_revision:
        previous_draft = state.get("final_answer", "")
        critique = state.get("critique", "")
        user_prompt = (
            f"{facts_context}\n\n"
            f"Question: {query}\n\n"
            f"Previous draft: {previous_draft}\n\n"
            f"Critic's feedback: {critique}\n\n"
            "Revised answer:"
        )
        system_prompt = _REVISION_SYSTEM_PROMPT
    else:
        user_prompt = f"{facts_context}\n\nQuestion: {query}\n\nAnswer:"
        system_prompt = _SYSTEM_PROMPT

    llm = get_llm_client()
    result = await llm.complete(system_prompt, user_prompt, max_tokens=600, temperature=0.3)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    final_answer = result.text.strip()

    logger.info(
        "writer_node — answer_length=%d revision=%s model=%s latency_ms=%d "
        "tokens_in=%s tokens_out=%s",
        len(final_answer),
        is_revision,
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
