"""Tests for the FactChecker agent (app/agents/fact_checker.py).

As of Day 10, FactChecker is verification-only: run() returns a 'facts' list but
does NOT set 'final_answer'. That responsibility belongs to Writer.
"""

import json
import logging
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.agents.fact_checker import run as fact_checker_run
from app.llm.client import LLMResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_search_result(n: int) -> dict:
    return {
        "title": f"Source {n}",
        "url": f"https://example.com/source-{n}",
        "content": f"Content of source {n}.",
        "score": 0.9,
    }


def _make_state(query: str = "Test question?", num_sources: int = 2) -> dict:
    return {
        "query": query,
        "search_results": [_make_search_result(i) for i in range(1, num_sources + 1)],
    }


def _llm_result(text: str) -> LLMResult:
    return LLMResult(
        text=text,
        provider="groq",
        model="llama-3.3-70b-versatile",
        latency_ms=80,
        tokens_in=200,
        tokens_out=50,
    )


def _make_mock_llm(response_text: str) -> MagicMock:
    mock_llm = MagicMock()
    mock_llm.complete = AsyncMock(return_value=_llm_result(response_text))
    return mock_llm


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


async def test_fact_checker_extracts_valid_facts(monkeypatch):
    """Well-formed LLM JSON → facts list populated, telemetry present, no final_answer set."""
    fc_json = json.dumps(
        {
            "facts": [
                {"claim": "Paris is the capital of France.", "sources": [1]},
                {"claim": "France is in Western Europe.", "sources": [1, 2]},
            ],
        }
    )
    monkeypatch.setattr("app.agents.fact_checker.get_llm_client", lambda: _make_mock_llm(fc_json))

    result = await fact_checker_run(_make_state())

    assert len(result["facts"]) == 2
    assert result["facts"][0]["claim"] == "Paris is the capital of France."
    assert result["facts"][0]["sources"] == [1]
    assert result["facts"][1]["sources"] == [1, 2]
    # Writer is responsible for final_answer — FactChecker must not set it.
    assert "final_answer" not in result
    assert result["provider"] == "groq"
    assert result["model"] == "llama-3.3-70b-versatile"
    assert result["tokens_in"] == 200
    assert result["tokens_out"] == 50


async def test_fact_checker_falls_back_on_invalid_json(monkeypatch, caplog):
    """Non-JSON LLM response → empty facts, no final_answer, WARNING logged."""
    monkeypatch.setattr(
        "app.agents.fact_checker.get_llm_client", lambda: _make_mock_llm("not json at all")
    )

    with caplog.at_level(logging.WARNING, logger="app.agents.fact_checker"):
        result = await fact_checker_run(_make_state())

    assert result["facts"] == []
    assert "final_answer" not in result
    assert any("JSON parse failure" in r.message for r in caplog.records)


async def test_fact_checker_falls_back_on_schema_mismatch(monkeypatch, caplog):
    """Valid JSON but missing required 'facts' key → empty facts, no final_answer, WARNING."""
    bad_json = json.dumps({"answer": "Some answer without facts key."})
    monkeypatch.setattr(
        "app.agents.fact_checker.get_llm_client", lambda: _make_mock_llm(bad_json)
    )

    with caplog.at_level(logging.WARNING, logger="app.agents.fact_checker"):
        result = await fact_checker_run(_make_state())

    assert result["facts"] == []
    assert "final_answer" not in result
    assert any("schema mismatch" in r.message for r in caplog.records)


async def test_fact_checker_drops_malformed_facts_keeps_valid_ones(monkeypatch):
    """Mixed valid/malformed facts → only valid facts survive, malformed are dropped."""
    fc_json = json.dumps(
        {
            "facts": [
                {"claim": "Valid fact with good sources.", "sources": [1]},
                {"sources": [2]},                          # missing "claim"
                {"claim": "", "sources": [1]},             # empty claim
                {"claim": "Non-int sources.", "sources": ["one"]},  # non-int sources
                {"claim": "Another valid fact.", "sources": [2]},
            ],
        }
    )
    monkeypatch.setattr("app.agents.fact_checker.get_llm_client", lambda: _make_mock_llm(fc_json))

    result = await fact_checker_run(_make_state(num_sources=2))

    assert len(result["facts"]) == 2
    claims = [f["claim"] for f in result["facts"]]
    assert "Valid fact with good sources." in claims
    assert "Another valid fact." in claims


async def test_fact_checker_filters_out_of_range_source_ids(monkeypatch):
    """Source ID beyond the actual results count is stripped; fact with no valid sources dropped."""
    fc_json = json.dumps(
        {
            "facts": [
                # source [9] is out of range when there are only 3 sources
                {"claim": "Hallucinated source.", "sources": [9]},
                # source [2] is valid; [9] is stripped, leaving [2] only
                {"claim": "Mixed fact.", "sources": [2, 9]},
                # fully in-range fact
                {"claim": "Good fact.", "sources": [1, 3]},
            ],
        }
    )
    monkeypatch.setattr("app.agents.fact_checker.get_llm_client", lambda: _make_mock_llm(fc_json))

    result = await fact_checker_run(_make_state(num_sources=3))

    # "Hallucinated source." has only [9] → no valid sources → dropped
    claims = [f["claim"] for f in result["facts"]]
    assert "Hallucinated source." not in claims

    # "Mixed fact." had [2, 9] → [9] stripped → keeps [2]
    mixed = next(f for f in result["facts"] if f["claim"] == "Mixed fact.")
    assert mixed["sources"] == [2]
    assert 9 not in mixed["sources"]

    # "Good fact." fully valid
    good = next(f for f in result["facts"] if f["claim"] == "Good fact.")
    assert good["sources"] == [1, 3]

    assert len(result["facts"]) == 2
