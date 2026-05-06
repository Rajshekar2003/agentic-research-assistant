"""Tests for the Writer agent (app/agents/writer.py)."""

import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.writer import _FALLBACK_ANSWER, run as writer_run
from app.llm.client import LLMResult, LLMUnavailableError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _llm_result(text: str) -> LLMResult:
    return LLMResult(
        text=text,
        provider="groq",
        model="llama-3.3-70b-versatile",
        latency_ms=60,
        tokens_in=150,
        tokens_out=40,
    )


def _make_mock_llm(response_text: str) -> MagicMock:
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(return_value=_llm_result(response_text))
    return mock_llm


def _make_state(query: str = "What is climate policy?", facts: list | None = None) -> dict:
    if facts is None:
        facts = [
            {"claim": "Brazil adopted a climate policy in 2023.", "sources": [1]},
            {"claim": "India's policy targets net-zero by 2070.", "sources": [2]},
        ]
    return {"query": query, "facts": facts}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_writer_synthesizes_from_facts(monkeypatch):
    """Non-empty facts → LLM called, final_answer set to stripped response, telemetry populated."""
    answer_text = "Brazil's policy is X [1]. India's policy is Y [2]."
    monkeypatch.setattr("app.agents.writer.get_llm_client", lambda: _make_mock_llm(answer_text))

    result = await writer_run(_make_state())

    assert result["final_answer"] == answer_text
    assert result["provider"] == "groq"
    assert result["model"] == "llama-3.3-70b-versatile"
    assert result["tokens_in"] == 150
    assert result["tokens_out"] == 40


async def test_writer_skips_llm_when_no_facts(monkeypatch, caplog):
    """Empty facts → LLM never called, fallback message returned, log records skipped=true."""
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(
        side_effect=AssertionError("Writer must not call LLM when facts are empty")
    )
    monkeypatch.setattr("app.agents.writer.get_llm_client", lambda: mock_llm)

    state = {"query": "What?", "facts": []}

    with caplog.at_level(logging.INFO, logger="app.agents.writer"):
        result = await writer_run(state)

    assert result["final_answer"] == _FALLBACK_ANSWER
    assert result["provider"] is None
    assert result["model"] is None
    assert result["tokens_in"] is None
    assert result["tokens_out"] is None
    assert any("skipped=true" in r.message for r in caplog.records)


async def test_writer_passes_facts_to_llm_in_prompt(monkeypatch):
    """The user prompt sent to complete() must contain all fact claims and the original query."""
    answer_text = "Climate policy varies. [1][2]"
    mock_llm = _make_mock_llm(answer_text)
    monkeypatch.setattr("app.agents.writer.get_llm_client", lambda: mock_llm)

    facts = [
        {"claim": "Fact one about climate.", "sources": [1]},
        {"claim": "Fact two about policy.", "sources": [2, 3]},
    ]
    query = "What is climate policy?"
    state = {"query": query, "facts": facts}

    await writer_run(state)

    call_args = mock_llm.complete.call_args
    user_prompt = call_args.args[1]  # second positional arg: complete(system, user, ...)

    assert "Fact one about climate." in user_prompt
    assert "Fact two about policy." in user_prompt
    assert query in user_prompt


async def test_writer_propagates_llm_failure(monkeypatch):
    """LLMUnavailableError from complete() propagates — endpoint catches it, not Writer."""
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(side_effect=LLMUnavailableError("Both providers failed."))
    monkeypatch.setattr("app.agents.writer.get_llm_client", lambda: mock_llm)

    with pytest.raises(LLMUnavailableError):
        await writer_run(_make_state())


async def test_writer_strips_whitespace_from_answer(monkeypatch):
    """Leading/trailing whitespace and newlines in the LLM response are stripped."""
    raw_text = "  answer with leading/trailing whitespace  \n"
    monkeypatch.setattr("app.agents.writer.get_llm_client", lambda: _make_mock_llm(raw_text))

    result = await writer_run(_make_state())

    assert result["final_answer"] == "answer with leading/trailing whitespace"
