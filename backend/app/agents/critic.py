"""Critic agent for the multi-agent research graph.

Day 11 — Critic evaluates the Writer's draft against verified facts and routes the graph.

Failure mode: if the Critic's LLM returns unparseable JSON or an invalid verdict, it falls
back to verdict="approve", shipping the Writer's most recent draft.  Skipping verification
is no worse than having no Critic at all; an infinite loop would be catastrophically worse.
LLMUnavailableError from llm.complete() is NOT caught here — it propagates to the endpoint
which returns 503.  Only parse/validation failures are silenced.

Input fields:  state["query"] (str), state["facts"] (list[dict]),
               state["final_answer"] (str), state.get("revision_count", 0) (int)
Output fields: state["critic_verdict"] (Literal["approve", "revise"]),
               state["critique"] (str — empty on approve, concrete instruction on revise),
               state["revision_count"] (int — incremented by 1 ONLY on "revise"),
               plus LLM telemetry (provider, model, tokens_in, tokens_out).
"""

import json
import logging
import time

from app.graph.state import ResearchState
from app.llm.client import get_llm_client

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a research critic. Given a user's question, a list of verified facts that were "
    "provided to a writer, and the writer's draft answer, evaluate whether the draft is "
    "acceptable for delivery to the user.\n\n"
    "Evaluate against these criteria:\n"
    "1. Groundedness: Every claim in the draft must be supported by the verified facts. The "
    "draft must not introduce new information not in the facts.\n"
    "2. Citations: The draft must use [1], [2], etc. inline citations matching the source IDs "
    "in the facts. Each significant claim should be cited.\n"
    "3. Completeness: The draft must address what the user actually asked. A draft that ignores "
    "key parts of the question is not acceptable.\n"
    "4. Clarity: The draft should be readable, concise, and not contradict itself.\n\n"
    "BE CALIBRATED. Approve drafts that are good enough — do not nitpick minor stylistic issues. "
    "Revise only when there is a concrete, fixable problem (e.g., uncited claim, missing aspect "
    "of the question, contradiction). The goal is publishable quality, not perfection.\n\n"
    'Output format — return a single JSON object with this exact shape and nothing else '
    "(no markdown fences, no preamble):\n\n"
    "{\n"
    '  "verdict": "approve" or "revise",\n'
    '  "critique": "if approve, empty string. If revise, a SPECIFIC instruction the writer '
    "can act on, e.g., 'Claim X is not supported by any fact; remove it or cite the supporting "
    "source.' Do not say 'improve clarity' — be concrete.\"\n"
    "}"
)


def _build_facts_block(facts: list[dict]) -> str:
    """Format verified facts as a numbered block for the Critic prompt.

    Args:
        facts: List of fact dicts, each with 'claim' (str) and 'sources' (list[int]).

    Returns:
        Multi-line string listing each fact with supporting source IDs, or a
        '(none)' placeholder when the list is empty.
    """
    if not facts:
        return "Verified facts: (none)"
    lines = ["Verified facts:"]
    for i, fact in enumerate(facts, start=1):
        src_ids = ", ".join(str(s) for s in fact.get("sources", []))
        lines.append(f"{i}. {fact['claim']} (sources: [{src_ids}])")
    return "\n".join(lines)


def _parse_verdict(raw: str) -> tuple[str, str]:
    """Parse and validate the LLM's JSON verdict response.

    Applies three layers of validation:
    1. JSON parse — falls back to approve on JSONDecodeError.
    2. Schema check — verdict must be exactly "approve" or "revise".
    3. Consistency check — revise with an empty critique is meaningless; force approve.

    Args:
        raw: Raw text returned by the LLM.

    Returns:
        Tuple of (verdict, critique). Always returns a valid ("approve"|"revise", str) pair.
    """
    try:
        parsed = json.loads(raw.strip())
    except json.JSONDecodeError:
        logger.warning("critic: JSON parse failure — raw: %.300s", raw)
        return "approve", ""

    if not isinstance(parsed, dict):
        logger.warning("critic: expected dict, got %s — raw: %.300s", type(parsed).__name__, raw)
        return "approve", ""

    verdict = parsed.get("verdict")
    critique = parsed.get("critique", "")

    if verdict not in ("approve", "revise"):
        logger.warning("critic: invalid verdict %r — falling back to approve", verdict)
        return "approve", ""

    if not isinstance(critique, str):
        critique = ""

    if verdict == "revise" and not critique.strip():
        logger.warning("critic: verdict=revise but critique is empty — forcing approve")
        return "approve", ""

    return verdict, critique


async def run(state: ResearchState) -> dict:
    """Evaluate the Writer's draft and decide whether to approve or request revision.

    Calls the LLM with a structured evaluation prompt.  On JSON parse failure or
    validation error, falls back to verdict='approve' (no revision) — intentional
    safety valve that prevents infinite loops if the Critic produces bad output.
    LLMUnavailableError propagates to the endpoint.

    revision_count is incremented by 1 ONLY when verdict == "revise".  The conditional
    edge in workflow.py reads this value to enforce the 2-revision hard cap.

    Args:
        state: ResearchState with 'query', 'facts', 'final_answer', and optionally
               'revision_count' (defaults to 0 when absent — first Critic call).

    Returns:
        State delta with 'critic_verdict' ("approve"|"revise"), 'critique' (str),
        'revision_count' (incremented only on revise), and LLM telemetry fields
        (provider, model, tokens_in, tokens_out).

    Raises:
        LLMUnavailableError: If both Groq and Gemini fail.  Propagates to the endpoint.
    """
    query = state["query"]
    facts = state.get("facts") or []
    final_answer = state.get("final_answer", "")
    revision_count = state.get("revision_count", 0)

    facts_block = _build_facts_block(facts)
    user_prompt = (
        f"Question: {query}\n\n"
        f"{facts_block}\n\n"
        f"Writer's draft: {final_answer}"
    )

    t0 = time.perf_counter()
    llm = get_llm_client()
    result = await llm.complete(_SYSTEM_PROMPT, user_prompt, max_tokens=400, temperature=0.2)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    verdict, critique = _parse_verdict(result.text)

    new_revision_count = revision_count + 1 if verdict == "revise" else revision_count

    logger.info(
        "critic_node — verdict=%s revision_count=%d model=%s latency_ms=%d "
        "tokens_in=%s tokens_out=%s",
        verdict,
        new_revision_count,
        result.model,
        latency_ms,
        result.tokens_in,
        result.tokens_out,
    )

    return {
        "critic_verdict": verdict,
        "critique": critique,
        "revision_count": new_revision_count,
        "provider": result.provider,
        "model": result.model,
        "tokens_in": result.tokens_in,
        "tokens_out": result.tokens_out,
    }
