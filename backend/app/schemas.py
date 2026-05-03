from pydantic import BaseModel, Field


class Source(BaseModel):
    title: str
    url: str
    snippet: str


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=3, max_length=500)


class ResearchResponse(BaseModel):
    answer: str
    sources: list[Source]
    elapsed_ms: int
