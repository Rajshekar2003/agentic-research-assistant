from fastapi import APIRouter
from app.schemas import ResearchRequest, ResearchResponse, Source

router = APIRouter(prefix="/research")

# TODO: replace stub with dispatcher that routes to baseline or LangGraph path


@router.post("")
def run_research(request: ResearchRequest) -> ResearchResponse:
    return ResearchResponse(
        answer="Research pipeline not yet implemented.",
        sources=[
            Source(
                title="Placeholder",
                url="https://example.com",
                snippet="This is a stub response.",
            )
        ],
        elapsed_ms=0,
    )
