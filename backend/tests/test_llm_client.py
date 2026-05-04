"""Tests for app.llm.client — LLMClient with Groq primary and Gemini fallback."""

import logging
from unittest.mock import AsyncMock, MagicMock

import groq as groq_module
import httpx
import pytest

from app.llm.client import LLMClient, LLMResult, LLMUnavailableError, get_llm_client


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_groq_response(text: str = "Paris is the capital of France.") -> MagicMock:
    usage = MagicMock()
    usage.prompt_tokens = 10
    usage.completion_tokens = 20
    message = MagicMock()
    message.content = text
    choice = MagicMock()
    choice.message = message
    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


def _make_gemini_response(text: str = "Gemini answer.") -> MagicMock:
    meta = MagicMock()
    meta.prompt_token_count = 8
    meta.candidates_token_count = 15
    resp = MagicMock()
    resp.text = text
    resp.usage_metadata = meta
    return resp


def _make_rate_limit_error() -> groq_module.RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/v1/chat/completions")
    response = httpx.Response(429, request=request, content=b'{"error": "rate limited"}')
    return groq_module.RateLimitError("Rate limit exceeded", response=response, body=None)


def _patch_clients(monkeypatch, *, groq_side_effect=None, groq_return=None, gemini_side_effect=None, gemini_return=None):
    """Wire up monkeypatched AsyncGroq and genai.Client on app.llm.client."""
    mock_groq_create = AsyncMock(side_effect=groq_side_effect, return_value=groq_return)
    mock_groq_instance = MagicMock()
    mock_groq_instance.chat.completions.create = mock_groq_create
    MockAsyncGroq = MagicMock(return_value=mock_groq_instance)
    monkeypatch.setattr("app.llm.client.AsyncGroq", MockAsyncGroq)

    mock_gemini_generate = AsyncMock(side_effect=gemini_side_effect, return_value=gemini_return)
    mock_gemini_instance = MagicMock()
    mock_gemini_instance.aio.models.generate_content = mock_gemini_generate
    mock_genai = MagicMock()
    mock_genai.Client = MagicMock(return_value=mock_gemini_instance)
    monkeypatch.setattr("app.llm.client.genai", mock_genai)

    return mock_groq_create, mock_gemini_generate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_llm_cache():
    """Ensure get_llm_client() lru_cache is clean for every test."""
    get_llm_client.cache_clear()
    yield
    get_llm_client.cache_clear()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_llm_client_groq_success(monkeypatch):
    """Successful Groq call populates all LLMResult fields with provider='groq'."""
    groq_resp = _make_groq_response("The capital of France is Paris.")
    _patch_clients(monkeypatch, groq_return=groq_resp)

    client = LLMClient(groq_api_key="test-groq", google_api_key="test-google")
    result = await client.complete("System prompt", "What is the capital of France?")

    assert isinstance(result, LLMResult)
    assert result.provider == "groq"
    assert result.text == "The capital of France is Paris."
    assert result.model == "llama-3.3-70b-versatile"
    assert result.latency_ms >= 0
    assert result.tokens_in == 10
    assert result.tokens_out == 20


async def test_llm_client_falls_back_to_gemini_on_rate_limit(monkeypatch, caplog):
    """RateLimitError from Groq triggers Gemini fallback and logs the event."""
    rate_limit_exc = _make_rate_limit_error()
    gemini_resp = _make_gemini_response("Fallback answer from Gemini.")

    _patch_clients(monkeypatch, groq_side_effect=rate_limit_exc, gemini_return=gemini_resp)

    client = LLMClient(groq_api_key="test-groq", google_api_key="test-google")

    with caplog.at_level(logging.WARNING, logger="app.llm.client"):
        result = await client.complete("System prompt", "Test query")

    assert result.provider == "gemini"
    assert result.text == "Fallback answer from Gemini."
    assert result.model == "gemini-2.5-flash-lite"
    assert result.latency_ms >= 0
    assert "RateLimitError" in caplog.text


async def test_llm_client_raises_when_both_fail(monkeypatch):
    """LLMUnavailableError is raised when both Groq and Gemini fail."""
    _patch_clients(
        monkeypatch,
        groq_side_effect=Exception("groq down"),
        gemini_side_effect=Exception("gemini down"),
    )

    client = LLMClient(groq_api_key="test-groq", google_api_key="test-google")

    with pytest.raises(LLMUnavailableError) as exc_info:
        await client.complete("System prompt", "Test query")

    message = str(exc_info.value)
    assert "Groq: Exception" in message
    assert "Gemini: Exception" in message


async def test_llm_client_singleton(monkeypatch):
    """get_llm_client() returns the same instance on repeated calls."""
    MockAsyncGroq = MagicMock(return_value=MagicMock())
    monkeypatch.setattr("app.llm.client.AsyncGroq", MockAsyncGroq)
    mock_genai = MagicMock()
    monkeypatch.setattr("app.llm.client.genai", mock_genai)

    instance_a = get_llm_client()
    instance_b = get_llm_client()

    assert instance_a is instance_b
    assert MockAsyncGroq.call_count == 1
