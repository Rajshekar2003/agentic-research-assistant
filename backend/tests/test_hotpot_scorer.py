"""Tests for backend/eval/hotpot/scorer.py.

The scorer's correctness is the foundation of every metric in the writeup.
All tests are pure-function tests on synthetic data — no mocking needed.
"""

import json
import math

import pytest

from eval.hotpot.scorer import (
    QuestionScores,
    exact_match_score,
    f1_score,
    normalize_answer,
    score_question,
    score_run,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_line(
    *,
    question_id: str = "q1",
    question_type: str = "comparison",
    question_level: str = "hard",
    ground_truth: str = "paris",
    baseline_status: str = "ok",
    baseline_answer: str | None = "paris",
    graph_status: str = "ok",
    graph_answer: str | None = "paris",
) -> dict:
    """Build a minimal JSONL-style dict matching the runner's output schema."""
    return {
        "question_id": question_id,
        "question_text": "What is the capital of France?",
        "ground_truth_answer": ground_truth,
        "question_type": question_type,
        "question_level": question_level,
        "baseline": {
            "status": baseline_status,
            "http_status": 200 if baseline_status == "ok" else None,
            "elapsed_ms": 1000,
            "answer": baseline_answer if baseline_status == "ok" else None,
            "sources": None,
            "facts": None,
            "critic_verdict": None,
            "revisions": None,
            "error": None if baseline_status == "ok" else "simulated error",
        },
        "graph": {
            "status": graph_status,
            "http_status": 200 if graph_status == "ok" else None,
            "elapsed_ms": 2000,
            "answer": graph_answer if graph_status == "ok" else None,
            "sources": None,
            "facts": None,
            "critic_verdict": None,
            "revisions": None,
            "error": None if graph_status == "ok" else "simulated error",
        },
        "timestamp": "2026-05-10T00:00:00+00:00",
    }


def _write_jsonl(tmp_path, lines: list[dict], filename: str = "run.jsonl"):
    path = tmp_path / filename
    path.write_text("\n".join(json.dumps(l) for l in lines) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# normalize_answer
# ---------------------------------------------------------------------------


def test_normalize_answer_lowercases():
    assert normalize_answer("Apple") == "apple"


def test_normalize_answer_strips_articles():
    assert normalize_answer("the apple") == "apple"
    assert normalize_answer("an apple") == "apple"
    assert normalize_answer("a apple") == "apple"


def test_normalize_answer_strips_punctuation():
    assert normalize_answer("Apple, Inc.") == "apple inc"


def test_normalize_answer_collapses_whitespace():
    assert normalize_answer("  apple   pie  ") == "apple pie"


def test_normalize_answer_combined():
    assert normalize_answer("The Apple, Inc.!") == "apple inc"


def test_normalize_answer_empty_string():
    assert normalize_answer("") == ""


def test_normalize_answer_only_articles():
    # "the a an" → all tokens are articles → collapses to empty string.
    assert normalize_answer("the a an") == ""


# ---------------------------------------------------------------------------
# exact_match_score
# ---------------------------------------------------------------------------


def test_exact_match_identical_strings():
    assert exact_match_score("apple", "apple") == 1


def test_exact_match_case_insensitive():
    assert exact_match_score("Apple", "apple") == 1


def test_exact_match_articles_dont_matter():
    assert exact_match_score("the apple", "an apple") == 1


def test_exact_match_substring_no_match():
    # EM requires full normalized equality, not substring containment.
    assert exact_match_score("apple pie", "apple") == 0


def test_exact_match_completely_different():
    assert exact_match_score("apple", "orange") == 0


def test_exact_match_empty_vs_nonempty():
    assert exact_match_score("", "apple") == 0


def test_exact_match_both_empty():
    assert exact_match_score("", "") == 1


# ---------------------------------------------------------------------------
# f1_score
# ---------------------------------------------------------------------------


def test_f1_perfect_overlap():
    f1, precision, recall = f1_score("apple pie", "apple pie")
    assert f1 == 1.0


def test_f1_no_overlap():
    f1, precision, recall = f1_score("apple", "orange")
    assert f1 == 0.0


def test_f1_partial_overlap_known_value():
    # prediction="apple pie", truth="pie crust"
    # tokens: ["apple","pie"] vs ["pie","crust"]
    # intersection: {"pie": 1} → num_same=1
    # precision=0.5, recall=0.5, f1=0.5
    f1, precision, recall = f1_score("apple pie", "pie crust")
    assert math.isclose(f1, 0.5)
    assert math.isclose(precision, 0.5)
    assert math.isclose(recall, 0.5)


def test_f1_normalization_applied():
    # "The Apple Pie" → "apple pie", "an Apple Pie!" → "apple pie"
    f1, _, _ = f1_score("The Apple Pie", "an Apple Pie!")
    assert f1 == 1.0


def test_f1_returns_tuple_of_three():
    result = f1_score("a", "a")
    assert len(result) == 3
    f1, precision, recall = result
    assert isinstance(f1, float)
    assert isinstance(precision, float)
    assert isinstance(recall, float)


def test_f1_both_empty_returns_one():
    # Both strings normalize to "" → vacuous match.
    f1, precision, recall = f1_score("", "")
    assert f1 == 1.0
    assert precision == 1.0
    assert recall == 1.0


def test_f1_one_side_empty_returns_zero():
    f1, precision, recall = f1_score("", "apple")
    assert f1 == 0.0
    f1, precision, recall = f1_score("apple", "")
    assert f1 == 0.0


# ---------------------------------------------------------------------------
# score_question
# ---------------------------------------------------------------------------


def test_score_question_both_ok():
    data = _make_line(ground_truth="paris", baseline_answer="paris", graph_answer="paris")
    scores = score_question(data)
    assert scores.baseline_em == 1
    assert scores.graph_em == 1
    assert math.isclose(scores.baseline_f1, 1.0)
    assert math.isclose(scores.graph_f1, 1.0)


def test_score_question_baseline_ok_graph_error():
    data = _make_line(
        ground_truth="paris",
        baseline_status="ok",
        baseline_answer="paris",
        graph_status="http_error",
        graph_answer=None,
    )
    scores = score_question(data)
    # Baseline scored.
    assert scores.baseline_em == 1
    assert scores.baseline_f1 is not None
    assert scores.baseline_status == "ok"
    # Graph not scored.
    assert scores.graph_em is None
    assert scores.graph_f1 is None
    assert scores.graph_status == "http_error"
    assert scores.graph_prediction is None


def test_score_question_extracts_question_metadata():
    data = _make_line(
        question_id="abc123",
        question_type="bridge",
        question_level="hard",
        ground_truth="tokyo",
        baseline_answer="tokyo",
        graph_answer="tokyo",
    )
    scores = score_question(data)
    assert scores.question_id == "abc123"
    assert scores.question_type == "bridge"
    assert scores.question_level == "hard"
    assert scores.ground_truth == "tokyo"


# ---------------------------------------------------------------------------
# score_run
# ---------------------------------------------------------------------------


def test_score_run_aggregates_means(tmp_path):
    # Design: baseline EMs=[1,1,0,0], F1s=[1.0,1.0,0.5,0.0]
    # Mean EM = 0.5, Mean F1 = 2.5/4 = 0.625
    lines = [
        _make_line(question_id="q1", ground_truth="apple pie", baseline_answer="apple pie"),
        _make_line(question_id="q2", ground_truth="orange", baseline_answer="orange"),
        # "apple crust" vs "apple pie": 1 common token ("apple"), 2 tokens each → f1=0.5
        _make_line(question_id="q3", ground_truth="apple pie", baseline_answer="apple crust"),
        _make_line(question_id="q4", ground_truth="apple", baseline_answer="orange"),
    ]
    path = _write_jsonl(tmp_path, lines)
    result = score_run(path)

    summary = result["summary"]
    assert summary["n_questions"] == 4
    assert summary["n_baseline_ok"] == 4
    assert math.isclose(summary["baseline_em"], 0.5)
    assert math.isclose(summary["baseline_f1"], 0.625)


def test_score_run_breaks_down_by_type(tmp_path):
    lines = [
        _make_line(question_id="q1", question_type="comparison", ground_truth="yes",
                   baseline_answer="yes", graph_answer="yes"),
        _make_line(question_id="q2", question_type="comparison", ground_truth="no",
                   baseline_answer="no", graph_answer="no"),
        _make_line(question_id="q3", question_type="bridge", ground_truth="paris",
                   baseline_answer="paris", graph_answer="paris"),
        _make_line(question_id="q4", question_type="bridge", ground_truth="tokyo",
                   baseline_answer="tokyo", graph_answer="tokyo"),
    ]
    path = _write_jsonl(tmp_path, lines)
    result = score_run(path)

    by_type = result["by_type"]
    assert "comparison" in by_type
    assert "bridge" in by_type
    assert by_type["comparison"]["n_questions"] == 2
    assert by_type["bridge"]["n_questions"] == 2
    assert math.isclose(by_type["comparison"]["baseline_em"], 1.0)
    assert math.isclose(by_type["bridge"]["baseline_em"], 1.0)


def test_score_run_skips_errored_questions_in_means(tmp_path):
    lines = [
        _make_line(question_id="q1", ground_truth="paris", baseline_answer="paris"),
        _make_line(question_id="q2", ground_truth="tokyo",
                   baseline_status="exception", baseline_answer=None),
    ]
    path = _write_jsonl(tmp_path, lines)
    result = score_run(path)

    summary = result["summary"]
    assert summary["n_questions"] == 2
    assert summary["n_baseline_ok"] == 1
    # Mean only over the 1 ok question: paris == paris → EM=1.0
    assert math.isclose(summary["baseline_em"], 1.0)


def test_score_run_per_question_list(tmp_path):
    lines = [
        _make_line(question_id="q1", ground_truth="paris", baseline_answer="paris"),
        _make_line(question_id="q2", ground_truth="tokyo", baseline_answer="tokyo"),
    ]
    path = _write_jsonl(tmp_path, lines)
    result = score_run(path)

    per_q = result["per_question"]
    assert len(per_q) == 2
    ids = {q["question_id"] for q in per_q}
    assert ids == {"q1", "q2"}
    # Each entry is a dict (asdict of QuestionScores).
    assert all(isinstance(q, dict) for q in per_q)


def test_score_run_empty_type_bucket_is_present(tmp_path):
    # Only comparison questions — bridge bucket should still exist with 0s.
    lines = [
        _make_line(question_id="q1", question_type="comparison",
                   ground_truth="yes", baseline_answer="yes"),
    ]
    path = _write_jsonl(tmp_path, lines)
    result = score_run(path)

    assert "bridge" in result["by_type"]
    assert result["by_type"]["bridge"]["n_questions"] == 0


def test_score_run_all_errors_mean_is_zero(tmp_path):
    lines = [
        _make_line(question_id="q1", baseline_status="timeout", graph_status="timeout"),
        _make_line(question_id="q2", baseline_status="exception", graph_status="exception"),
    ]
    path = _write_jsonl(tmp_path, lines)
    result = score_run(path)

    summary = result["summary"]
    assert summary["n_baseline_ok"] == 0
    assert summary["n_graph_ok"] == 0
    assert summary["baseline_em"] == 0.0
    assert summary["graph_em"] == 0.0
