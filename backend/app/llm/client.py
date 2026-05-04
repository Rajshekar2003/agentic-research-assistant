"""
LLM client for the Agentic Research Assistant.

Design:
- Primary provider: Groq via AsyncGroq (model: llama-3.3-70b-versatile), 30 s per-call
  timeout configured on the SDK client itself.
- Fallback provider: Gemini via google-genai async client (model: gemini-2.5-flash-lite),
  30 s timeout enforced with asyncio.wait_for because the SDK does not expose a top-level
  timeout constructor argument in the same way.
- No retries before triggering fallback. The fallback IS the retry strategy. Retrying Groq
  on a rate-limit or 5xx just queues more load on a provider that is already struggling;
  switching providers immediately is the correct response.
- Exception taxonomy — triggers Groq → Gemini fallback:
    groq.RateLimitError        – quota exhausted (HTTP 429)
    groq.APIStatusError(>=500) – provider-side server failure
    groq.APIConnectionError    – network unreachable or SDK-level timeout
    asyncio.TimeoutError       – per-call 30 s budget exceeded
    Exception (catch-all)      – any other unexpected error (logged at WARNING)
  If Gemini also fails, LLMUnavailableError is raised with a sanitized message that
  names only the exception types — never raw provider error details.
"""

import asyncio
import logging
import time
from functools import lru_cache
from typing import Literal

import groq as groq_module
from google import genai
from google.genai import types as genai_types
from groq import AsyncGroq
from pydantic import BaseModel

from app.config import get_settings

logger = logging.getLogger(__name__)

_GROQ_MODEL = "llama-3.3-70b-versatile"
_GEMINI_MODEL = "gemini-2.5-flash-lite"
_TIMEOUT_SECONDS = 30.0


class LLMUnavailableError(Exception):
    """Raised when both Groq and Gemini have failed to produce a response.

    The message names only exception types, never raw provider error details, so it is
    safe to surface in structured logs and 503 responses without leaking internals.
    """


class LLMResult(BaseModel):
    """Successful result from a single LLMClient.complete() call.

    Args:
        text: The model-generated text.
        provider: Which provider produced the result.
        model: The specific model identifier used.
        latency_ms: Round-trip time in milliseconds for the LLM call only.
        tokens_in: Prompt/input token count, or None if the SDK did not report it.
        tokens_out: Completion/output token count, or None if the SDK did not report it.
    """

    text: str
    provider: Literal["groq", "gemini"]
    model: str
    latency_ms: int
    tokens_in: int | None
    tokens_out: int | None


class LLMClient:
    """Async LLM client with Groq as primary and Gemini as automatic fallback.

    Args:
        groq_api_key: Plaintext Groq API key (SecretStr already unwrapped by caller).
        google_api_key: Plaintext Google API key (SecretStr already unwrapped by caller).
    """

    def __init__(self, groq_api_key: str, google_api_key: str) -> None:
        self._groq = AsyncGroq(api_key=groq_api_key, timeout=_TIMEOUT_SECONDS)
        self._gemini = genai.Client(api_key=google_api_key)

    async def complete(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> LLMResult:
        """Generate a completion using Groq, with Gemini as automatic fallback.

        Args:
            system: System prompt that sets assistant behaviour and constraints.
            user: The user's query or message to respond to.
            max_tokens: Maximum tokens to generate in the response.
            temperature: Sampling temperature; 0.0 is deterministic, 1.0 is creative.

        Returns:
            LLMResult with generated text, provider, model, latency, and token counts.

        Raises:
            LLMUnavailableError: If both providers fail. The message names exception
                types only — raw provider details are never exposed to callers.
        """
        groq_error: Exception | None = None
        gemini_error: Exception | None = None

        # --- Primary: Groq ---
        try:
            return await self._call_groq(system, user, max_tokens, temperature)
        except groq_module.RateLimitError as exc:
            groq_error = exc
            logger.warning(
                "Groq fallback triggered by %s — trying Gemini",
                type(exc).__name__,
            )
        except groq_module.APIStatusError as exc:
            if exc.status_code >= 500:
                groq_error = exc
                logger.warning(
                    "Groq fallback triggered by APIStatusError(status=%d) — trying Gemini",
                    exc.status_code,
                )
            else:
                raise
        except groq_module.APIConnectionError as exc:
            groq_error = exc
            logger.warning(
                "Groq fallback triggered by %s — trying Gemini",
                type(exc).__name__,
            )
        except asyncio.TimeoutError as exc:
            groq_error = exc
            logger.warning(
                "Groq fallback triggered by TimeoutError after %.0fs — trying Gemini",
                _TIMEOUT_SECONDS,
            )
        except Exception as exc:  # noqa: BLE001
            groq_error = exc
            logger.warning(
                "Groq fallback triggered by unexpected %s — trying Gemini",
                type(exc).__name__,
            )

        # --- Fallback: Gemini ---
        try:
            return await self._call_gemini(system, user, max_tokens, temperature)
        except Exception as exc:  # noqa: BLE001
            gemini_error = exc
            logger.error(
                "Gemini fallback also failed with %s",
                type(exc).__name__,
            )

        raise LLMUnavailableError(
            f"Both LLM providers failed. "
            f"Groq: {type(groq_error).__name__}. "
            f"Gemini: {type(gemini_error).__name__}."
        )

    async def _call_groq(
        self, system: str, user: str, max_tokens: int, temperature: float
    ) -> LLMResult:
        """Call the Groq chat-completions API.

        Args:
            system: System prompt.
            user: User message.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.

        Returns:
            LLMResult populated from the Groq chat-completion response.
        """
        t0 = time.perf_counter()
        response = await self._groq.chat.completions.create(
            model=_GROQ_MODEL,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        text = response.choices[0].message.content or ""
        tokens_in = response.usage.prompt_tokens if response.usage else None
        tokens_out = response.usage.completion_tokens if response.usage else None
        return LLMResult(
            text=text,
            provider="groq",
            model=_GROQ_MODEL,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )

    async def _call_gemini(
        self, system: str, user: str, max_tokens: int, temperature: float
    ) -> LLMResult:
        """Call the Gemini generate-content API via the google-genai async client.

        Args:
            system: System prompt passed via GenerateContentConfig.
            user: User message content.
            max_tokens: Maximum output tokens.
            temperature: Sampling temperature.

        Returns:
            LLMResult populated from the Gemini generate-content response.
        """
        t0 = time.perf_counter()
        response = await asyncio.wait_for(
            self._gemini.aio.models.generate_content(
                model=_GEMINI_MODEL,
                contents=user,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system,
                    max_output_tokens=max_tokens,
                    temperature=temperature,
                ),
            ),
            timeout=_TIMEOUT_SECONDS,
        )
        latency_ms = int((time.perf_counter() - t0) * 1000)
        text = response.text or ""
        meta = response.usage_metadata
        tokens_in = meta.prompt_token_count if meta else None
        tokens_out = meta.candidates_token_count if meta else None
        return LLMResult(
            text=text,
            provider="gemini",
            model=_GEMINI_MODEL,
            latency_ms=latency_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
        )


@lru_cache
def get_llm_client() -> LLMClient:
    """Return the singleton LLMClient, constructing it once from application settings.

    SecretStr values are extracted via .get_secret_value() exactly once here and passed
    as plain strings to LLMClient.__init__. They are never forwarded to any logger call.

    Returns:
        The cached LLMClient instance.
    """
    settings = get_settings()
    return LLMClient(
        groq_api_key=settings.groq_api_key.get_secret_value(),
        google_api_key=settings.google_api_key.get_secret_value(),
    )
