"""Tests for the Planner agent (app/agents/planner.py)."""

import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.planner import run as planner_run
from app.llm.client import LLMResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_llm_result(text: str) -> LLMResult:
    return LLMResult(
        text=text,
        provider="groq",
        model="llama-3.3-70b-versatile",
        latency_ms=50,
        tokens_in=20,
        tokens_out=10,
    )


def _make_mock_llm(text: str) -> MagicMock:
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(return_value=_mock_llm_result(text))
    return mock_llm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_planner_returns_valid_plan_for_complex_query(monkeypatch):
    """Planner returns a 2-4 item plan when the LLM returns a valid JSON array."""
    plan_data = ["What caused WW2?", "When did WW2 end?", "Who were the main leaders?"]
    monkeypatch.setattr(
        "app.agents.planner.get_llm_client", lambda: _make_mock_llm(json.dumps(plan_data))
    )

    result = await planner_run({"query": "Tell me about World War 2"})

    plan = result["plan"]
    assert 1 <= len(plan) <= 4
    assert all(isinstance(s, str) and s for s in plan)
    assert all(s == s.strip() for s in plan)
    assert plan == plan_data


async def test_planner_falls_back_on_invalid_json(monkeypatch, caplog):
    """Planner falls back to [query] when the LLM returns non-JSON text."""
    monkeypatch.setattr(
        "app.agents.planner.get_llm_client", lambda: _make_mock_llm("this is not JSON")
    )
    original_query = "What is the speed of light?"

    with caplog.at_level(logging.WARNING, logger="app.agents.planner"):
        result = await planner_run({"query": original_query})

    assert result["plan"] == [original_query]
    assert any("JSON parse failure" in r.message for r in caplog.records)


async def test_planner_falls_back_on_wrong_schema(monkeypatch, caplog):
    """Planner falls back to [query] when the LLM returns a JSON object, not an array."""
    monkeypatch.setattr(
        "app.agents.planner.get_llm_client",
        lambda: _make_mock_llm('{"key": "value"}'),
    )
    original_query = "What is photosynthesis?"

    with caplog.at_level(logging.WARNING, logger="app.agents.planner"):
        result = await planner_run({"query": original_query})

    assert result["plan"] == [original_query]


async def test_planner_truncates_oversized_plan(monkeypatch):
    """Planner caps the plan at 4 items when the LLM returns more than 4 sub-questions."""
    big_plan = [f"sub-question {i}" for i in range(10)]
    monkeypatch.setattr(
        "app.agents.planner.get_llm_client", lambda: _make_mock_llm(json.dumps(big_plan))
    )

    result = await planner_run({"query": "What is everything about the universe?"})

    assert len(result["plan"]) == 4
    assert result["plan"] == big_plan[:4]
