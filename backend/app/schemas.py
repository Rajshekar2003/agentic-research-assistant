"""Shared Pydantic request/response models for the research assistant API."""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator


class Source(BaseModel):
    """A single search-result source attached to a research response.

    Args:
        title: Display title of the source page.
        url: Canonical URL of the source.
        snippet: Short excerpt from the source (max 2000 chars).
    """

    model_config = ConfigDict(str_strip_whitespace=True, populate_by_name=True)

    title: str = Field(..., min_length=1)
    url: HttpUrl
    snippet: str = Field(..., max_length=2000)


class ResearchRequest(BaseModel):
    """Incoming research query submitted by the client.

    Args:
        query: Natural-language research question (3–500 characters, must
               contain at least one word character after stripping).
    """

    model_config = ConfigDict(str_strip_whitespace=True, populate_by_name=True)

    query: str = Field(..., min_length=3, max_length=500)

    @model_validator(mode="after")
    def reject_punctuation_only(self) -> "ResearchRequest":
        """Reject queries that contain no word characters after whitespace stripping.

        Returns:
            The validated ResearchRequest instance.

        Raises:
            ValueError: If the stripped query has no alphanumeric/underscore characters.
        """
        if not re.search(r"\w", self.query):
            raise ValueError("Query must contain at least one word character.")
        return self


class ResearchResponse(BaseModel):
    """Response payload returned by the research pipeline.

    Args:
        answer: The synthesised answer text.
        sources: List of supporting sources (empty for stub/baseline).
        elapsed_ms: Wall-clock time spent processing the request, in milliseconds.
        mode: Which pipeline path produced this response.
    """

    model_config = ConfigDict(str_strip_whitespace=True, populate_by_name=True)

    answer: str
    sources: list[Source] = []
    elapsed_ms: int = Field(..., ge=0)
    mode: Literal["baseline", "graph"] = "baseline"


class HealthResponse(BaseModel):
    """Health-check response for the /health liveness probe.

    Args:
        status: Always "ok" when the app is healthy.
        version: Application version string.
        env: Active deployment environment.
    """

    model_config = ConfigDict(str_strip_whitespace=True, populate_by_name=True)

    status: Literal["ok"]
    version: str
    env: str
