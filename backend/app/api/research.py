"""Research endpoint — accepts a validated query and returns a model-generated answer."""

import logging
import time

from fastapi import APIRouter, HTTPException

from app.llm.client import LLMUnavailableError, get_llm_client
from app.schemas import ResearchRequest, ResearchResponse

router = APIRouter(prefix="/research")
logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = (
    "You are a research assistant. Answer the user's question concisely and accurately. "
    "If you are uncertain, say so. Do not fabricate facts. "
    "Do not include citations or [1] markers — those will be added by a later component."
)


@router.post("", response_model=ResearchResponse)
async def run_research(request: ResearchRequest) -> ResearchResponse:
    """Accept a validated research query and return a model-generated answer.

    # TODO Day 5 (Search): Call Tavily search tool, populate `sources` list with
    #   real results, and feed retrieved context into the LLM prompt.

    Args:
        request: Validated research request containing the query string.

    Returns:
        ResearchResponse with model-generated answer, empty sources list, wall-clock
        elapsed time in milliseconds, and mode set to "baseline".

    Raises:
        HTTPException: 503 if both LLM providers are unavailable.
    """
    start = time.perf_counter()
    logger.info("Research query received: %.100s", request.query)

    llm = get_llm_client()
    try:
        result = await llm.complete(_SYSTEM_PROMPT, request.query)
    except LLMUnavailableError:
        raise HTTPException(status_code=503, detail="Research service temporarily unavailable")

    elapsed_ms = int((time.perf_counter() - start) * 1000)

    logger.info(
        "Research complete: provider=%s model=%s latency_ms=%s tokens_in=%s tokens_out=%s "
        "total_elapsed_ms=%s",
        result.provider,
        result.model,
        result.latency_ms,
        result.tokens_in,
        result.tokens_out,
        elapsed_ms,
    )

    return ResearchResponse(
        answer=result.text,
        sources=[],
        elapsed_ms=elapsed_ms,
        mode="baseline",
    )
