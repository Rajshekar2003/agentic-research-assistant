"""Tests for the Critic agent (app/agents/critic.py).

The Critic is intentionally designed to never raise on parse/validation failures —
it falls back to verdict="approve" instead of propagating bad LLM output up the
call stack.  This is the safe failure mode: skipping verification is no worse than
having no Critic at all; an infinite revision loop would be catastrophically worse.

LLMUnavailableError from the LLM call itself is NOT caught by Critic and DOES
propagate — that is a different failure (the LLM is down) that the endpoint handles.
"""

import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.critic import run as critic_run
from app.llm.client import LLMResult, LLMUnavailableError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _llm_result(text: str) -> LLMResult:
    return LLMResult(
        text=text,
        provider="groq",
        model="llama-3.3-70b-versatile",
        latency_ms=80,
        tokens_in=200,
        tokens_out=30,
    )


def _make_mock_llm(response_text: str) -> MagicMock:
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(return_value=_llm_result(response_text))
    return mock_llm


def _approve_json() -> str:
    return json.dumps({"verdict": "approve", "critique": ""})


def _revise_json(critique: str = "Claim X is not supported by any fact; remove it.") -> str:
    return json.dumps({"verdict": "revise", "critique": critique})


def _base_state(revision_count: int = 0) -> dict:
    return {
        "query": "What is the capital of France?",
        "facts": [{"claim": "France's capital is Paris.", "sources": [1]}],
        "final_answer": "France's capital is Paris. [1]",
        "revision_count": revision_count,
    }


# ---------------------------------------------------------------------------
# Tests — happy paths
# ---------------------------------------------------------------------------


async def test_critic_approves_acceptable_draft(monkeypatch):
    """Approve verdict → delta has critic_verdict='approve', critique='', no revision_count change."""
    monkeypatch.setattr("app.agents.critic.get_llm_client", lambda: _make_mock_llm(_approve_json()))

    result = await critic_run(_base_state(revision_count=0))

    assert result["critic_verdict"] == "approve"
    assert result["critique"] == ""
    assert result["revision_count"] == 0  # not incremented on approve
    assert result["provider"] == "groq"
    assert result["model"] == "llama-3.3-70b-versatile"
    assert result["tokens_in"] == 200
    assert result["tokens_out"] == 30


async def test_critic_revises_with_feedback(monkeypatch):
    """Revise verdict → delta has critic_verdict='revise', specific critique, revision_count++."""
    critique_text = "Claim X is not supported by any fact; remove it or cite the source."
    monkeypatch.setattr(
        "app.agents.critic.get_llm_client",
        lambda: _make_mock_llm(_revise_json(critique_text)),
    )

    result = await critic_run(_base_state(revision_count=0))

    assert result["critic_verdict"] == "revise"
    assert result["critique"] == critique_text
    assert result["revision_count"] == 1  # incremented from 0
    assert result["provider"] == "groq"
    assert result["tokens_in"] == 200
    assert result["tokens_out"] == 30


# ---------------------------------------------------------------------------
# Tests — fallback-to-approve on parse/validation failures
# ---------------------------------------------------------------------------


async def test_critic_falls_back_to_approve_on_invalid_json(monkeypatch, caplog):
    """LLM returns non-JSON text → parse failure → approve fallback, WARNING logged.

    Safety rationale: a Critic that produces bad output should not block delivery.
    Falling back to approve ships the Writer's draft, which is no worse than having
    no Critic.  The WARNING in logs makes the failure diagnosable.
    """
    monkeypatch.setattr(
        "app.agents.critic.get_llm_client", lambda: _make_mock_llm("not valid JSON at all")
    )

    with caplog.at_level(logging.WARNING, logger="app.agents.critic"):
        result = await critic_run(_base_state())

    assert result["critic_verdict"] == "approve"
    assert result["critique"] == ""
    assert result["revision_count"] == 0
    assert any("JSON parse failure" in r.message for r in caplog.records)


async def test_critic_falls_back_to_approve_on_invalid_verdict(monkeypatch, caplog):
    """LLM returns valid JSON but verdict is not 'approve' or 'revise' → approve fallback."""
    bad_json = json.dumps({"verdict": "maybe", "critique": "some critique"})
    monkeypatch.setattr(
        "app.agents.critic.get_llm_client", lambda: _make_mock_llm(bad_json)
    )

    with caplog.at_level(logging.WARNING, logger="app.agents.critic"):
        result = await critic_run(_base_state())

    assert result["critic_verdict"] == "approve"
    assert result["critique"] == ""
    assert result["revision_count"] == 0
    assert any("invalid verdict" in r.message for r in caplog.records)


async def test_critic_forces_approve_when_revise_has_empty_critique(monkeypatch, caplog):
    """verdict='revise' with empty critique → forced to 'approve'.

    A revision instruction with no concrete feedback is actionless — the Writer
    cannot improve without knowing what to fix.  Approving is safer than looping.
    """
    bad_json = json.dumps({"verdict": "revise", "critique": ""})
    monkeypatch.setattr(
        "app.agents.critic.get_llm_client", lambda: _make_mock_llm(bad_json)
    )

    with caplog.at_level(logging.WARNING, logger="app.agents.critic"):
        result = await critic_run(_base_state())

    assert result["critic_verdict"] == "approve"
    assert result["critique"] == ""
    assert result["revision_count"] == 0
    assert any("forcing approve" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Tests — revision_count increment logic
# ---------------------------------------------------------------------------


async def test_critic_increments_revision_count_only_on_revise(monkeypatch):
    """revision_count is incremented only when verdict=='revise', not on 'approve'."""
    # Approve with initial rc=1 → stays at 1
    monkeypatch.setattr("app.agents.critic.get_llm_client", lambda: _make_mock_llm(_approve_json()))
    result_approve = await critic_run(_base_state(revision_count=1))
    assert result_approve["revision_count"] == 1

    # Revise with initial rc=0 → becomes 1
    monkeypatch.setattr(
        "app.agents.critic.get_llm_client", lambda: _make_mock_llm(_revise_json())
    )
    result_revise = await critic_run(_base_state(revision_count=0))
    assert result_revise["revision_count"] == 1


async def test_critic_propagates_llm_unavailable_error(monkeypatch):
    """LLMUnavailableError from llm.complete() propagates — endpoint catches it for 503."""
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(side_effect=LLMUnavailableError("Both providers failed."))
    monkeypatch.setattr("app.agents.critic.get_llm_client", lambda: mock_llm)

    with pytest.raises(LLMUnavailableError):
        await critic_run(_base_state())


async def test_critic_user_prompt_contains_all_sections(monkeypatch):
    """User prompt sent to the LLM includes question, facts block, and writer's draft."""
    mock_llm = _make_mock_llm(_approve_json())
    monkeypatch.setattr("app.agents.critic.get_llm_client", lambda: mock_llm)

    state = {
        "query": "What is quantum computing?",
        "facts": [{"claim": "Quantum computers use qubits.", "sources": [2]}],
        "final_answer": "Quantum computers use qubits. [2]",
        "revision_count": 0,
    }
    await critic_run(state)

    call_args = mock_llm.complete.call_args
    user_prompt = call_args.args[1]

    assert "What is quantum computing?" in user_prompt
    assert "Quantum computers use qubits." in user_prompt
    assert "Quantum computers use qubits. [2]" in user_prompt
