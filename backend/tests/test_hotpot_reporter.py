"""Tests for backend/eval/hotpot/reporter.py.

All tests are pure-function tests on synthetic data — no live endpoints, no file I/O
except for the output_path test which uses tmp_path.
"""

import json
from pathlib import Path

import pytest

from eval.hotpot.reporter import (
    format_pct,
    is_refusal,
    latency_stats,
    render_report,
    select_sample_questions,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _scored_q(
    *,
    question_id: str = "q1",
    question_type: str = "bridge",
    question_level: str = "hard",
    ground_truth: str = "answer",
    baseline_prediction: str | None = "answer",
    baseline_em: int | None = 1,
    baseline_f1: float | None = 1.0,
    graph_prediction: str | None = "answer",
    graph_em: int | None = 1,
    graph_f1: float | None = 1.0,
    baseline_status: str = "ok",
    graph_status: str = "ok",
) -> dict:
    """Build a minimal scored-question dict matching scorer.py's per_question schema."""
    return {
        "question_id": question_id,
        "question_type": question_type,
        "question_level": question_level,
        "ground_truth": ground_truth,
        "baseline_prediction": baseline_prediction,
        "baseline_em": baseline_em,
        "baseline_f1": baseline_f1,
        "graph_prediction": graph_prediction,
        "graph_em": graph_em,
        "graph_f1": graph_f1,
        "baseline_status": baseline_status,
        "graph_status": graph_status,
    }


def _minimal_scores(per_question: list[dict] | None = None) -> dict:
    """Build a minimal scores dict with the structure produced by scorer.py."""
    return {
        "summary": {
            "n_questions": 5,
            "n_baseline_ok": 5,
            "n_graph_ok": 5,
            "baseline_em": 0.0,
            "baseline_f1": 0.042,
            "graph_em": 0.0,
            "graph_f1": 0.037,
        },
        "by_type": {
            "comparison": {
                "n_questions": 1,
                "n_baseline_ok": 1,
                "n_graph_ok": 1,
                "baseline_em": 0.0,
                "baseline_f1": 0.077,
                "graph_em": 0.0,
                "graph_f1": 0.1,
            },
            "bridge": {
                "n_questions": 4,
                "n_baseline_ok": 4,
                "n_graph_ok": 4,
                "baseline_em": 0.0,
                "baseline_f1": 0.033,
                "graph_em": 0.0,
                "graph_f1": 0.021,
            },
        },
        "per_question": per_question if per_question is not None else [],
    }


def _run_entry(question_id: str, b_elapsed: int = 1000, g_elapsed: int = 3000) -> dict:
    """Build a minimal runner JSONL dict for a single question."""
    return {
        "question_id": question_id,
        "question_text": f"What is the answer to {question_id}?",
        "ground_truth_answer": "answer",
        "question_type": "bridge",
        "question_level": "hard",
        "baseline": {
            "status": "ok",
            "http_status": 200,
            "elapsed_ms": b_elapsed,
            "answer": "Some answer",
            "sources": [],
            "facts": None,
            "critic_verdict": None,
            "revisions": None,
            "error": None,
        },
        "graph": {
            "status": "ok",
            "http_status": 200,
            "elapsed_ms": g_elapsed,
            "answer": "Some answer",
            "sources": [],
            "facts": [{"claim": "c1", "sources": [1]}],
            "critic_verdict": "approve",
            "revisions": 0,
            "error": None,
        },
    }


# ---------------------------------------------------------------------------
# is_refusal tests
# ---------------------------------------------------------------------------


def test_is_refusal_matches_couldnt_verify():
    assert is_refusal("I couldn't verify any claims") is True


def test_is_refusal_matches_no_information():
    assert is_refusal("The sources do not contain information about this topic.") is True


def test_is_refusal_case_insensitive():
    assert is_refusal("I COULDN'T VERIFY the facts.") is True


def test_is_refusal_normal_answer_returns_false():
    assert is_refusal("The capital is Ulaanbaatar") is False


def test_is_refusal_empty_string():
    assert is_refusal("") is False


# ---------------------------------------------------------------------------
# select_sample_questions tests
# ---------------------------------------------------------------------------


def test_select_sample_picks_graph_winner():
    qs = [
        _scored_q(question_id="q1", graph_f1=0.8, baseline_f1=0.2, graph_em=0, baseline_em=0),
        _scored_q(question_id="q2", graph_f1=0.3, baseline_f1=0.3, graph_em=0, baseline_em=0),
        _scored_q(question_id="q3", graph_f1=0.1, baseline_f1=0.5, graph_em=0, baseline_em=0),
    ]
    selected = select_sample_questions(qs)
    ids = [q["question_id"] for q in selected]
    assert "q1" in ids, "q1 has the highest graph_f1 - baseline_f1 delta and should be selected"


def test_select_sample_handles_too_few_questions():
    qs = [
        _scored_q(question_id="q1", graph_f1=0.8, baseline_f1=0.2),
        _scored_q(question_id="q2", graph_f1=0.1, baseline_f1=0.6),
    ]
    selected = select_sample_questions(qs, n=4)
    assert len(selected) == 2, "Should return only as many as are available — no duplicates"
    ids = [q["question_id"] for q in selected]
    assert len(ids) == len(set(ids))


def test_select_sample_avoids_duplicates():
    # All questions have 0 F1, so the "graph winner" is also a both-zero candidate.
    # The algorithm must not select the same question twice.
    qs = [
        _scored_q(question_id=f"q{i}", graph_f1=0.0, baseline_f1=0.0, graph_em=0, baseline_em=0)
        for i in range(4)
    ]
    selected = select_sample_questions(qs, n=4)
    ids = [q["question_id"] for q in selected]
    assert len(ids) == len(set(ids)), "No question should appear more than once"


# ---------------------------------------------------------------------------
# latency_stats tests
# ---------------------------------------------------------------------------


def _make_run_data(b_ms_list: list[int], g_ms_list: list[int] | None = None) -> list[dict]:
    """Build synthetic run_data with the given baseline elapsed_ms values."""
    if g_ms_list is None:
        g_ms_list = [500] * len(b_ms_list)
    return [
        {
            "baseline": {"status": "ok", "elapsed_ms": b},
            "graph": {"status": "ok", "elapsed_ms": g},
        }
        for b, g in zip(b_ms_list, g_ms_list)
    ]


def test_latency_stats_computes_mean_median_p95():
    run_data = _make_run_data([100, 200, 300, 400, 500])
    stats = latency_stats(run_data)
    b = stats["baseline"]
    assert b["mean_ms"] == 300
    assert b["median_ms"] == 300
    # p95_idx = int(5 * 0.95) = 4; sorted[4] = 500
    assert b["p95_ms"] == 500


def test_latency_stats_skips_errored_entries():
    run_data = _make_run_data([100, 200, 300, 400, 500])
    # Add one errored baseline entry — should be excluded
    run_data.append({
        "baseline": {"status": "http_error", "elapsed_ms": 9999},
        "graph": {"status": "ok", "elapsed_ms": 500},
    })
    stats = latency_stats(run_data)
    # Only the 5 ok entries contribute; mean = (100+200+300+400+500)/5 = 300
    assert stats["baseline"]["mean_ms"] == 300


# ---------------------------------------------------------------------------
# render_report tests
# ---------------------------------------------------------------------------


def test_render_report_includes_headline_numbers():
    scores = _minimal_scores()
    result = render_report(scores)
    assert "Headline numbers" in result
    # summary baseline_f1=0.042, should appear as 0.042
    assert "0.042" in result
    # summary graph_f1=0.037
    assert "0.037" in result


def test_render_report_handles_empty_per_question():
    scores = _minimal_scores(per_question=[])
    result = render_report(scores)
    # Should not crash and should note the absence of sample cases
    assert "Only 0 distinct cases available in this run." in result


def test_render_report_omits_latency_when_no_run_jsonl():
    scores = _minimal_scores()
    result = render_report(scores, run_jsonl_path=None)
    # Placeholder text should appear when run JSONL is not provided
    assert "Latency from run JSONL" in result
    # No concrete millisecond figures should appear in the latency table
    assert "mean_ms" not in result


def test_render_report_writes_to_output_path(tmp_path: Path):
    scores = _minimal_scores()
    output = tmp_path / "report.md"
    result = render_report(scores, output_path=output)
    assert output.exists(), "Output file should be created"
    assert output.read_text(encoding="utf-8") == result
