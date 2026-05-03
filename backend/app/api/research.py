"""Research endpoint — accepts a validated query and returns a stub response."""

import logging
import time

from fastapi import APIRouter, Depends

from app.config import Settings, get_settings
from app.schemas import ResearchRequest, ResearchResponse

router = APIRouter(prefix="/research")
logger = logging.getLogger(__name__)


@router.post("", response_model=ResearchResponse)
def run_research(
    request: ResearchRequest,
    settings: Settings = Depends(get_settings),
) -> ResearchResponse:
    """Accept a validated research query and return a stub response.

    # TODO Day 4 (LLM): Route query through Groq/Gemini via langchain-groq /
    #   langchain-google-genai; replace stub answer with model-generated text.
    # TODO Day 5 (Search): Call Tavily search tool, populate `sources` list with
    #   real results, and feed retrieved context into the LLM prompt.

    Args:
        request: Validated research request containing the query string.
        settings: Application settings injected via FastAPI dependency.

    Returns:
        ResearchResponse with a stub answer, empty sources, wall-clock elapsed
        time in milliseconds, and mode set to "baseline".
    """
    start = time.perf_counter()

    logger.info("Research query received: %.100s", request.query)

    elapsed_ms = int((time.perf_counter() - start) * 1000)

    return ResearchResponse(
        answer=f"[stub] Received query: {request.query}",
        sources=[],
        elapsed_ms=elapsed_ms,
        mode="baseline",
    )
