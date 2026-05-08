"""Tests for the HotpotQA loader (eval/hotpot/loader.py).

All tests use local fixtures — no network calls are made.
"""

import json
import logging

import pytest

from eval.hotpot.loader import (
    HotpotQuestion,
    download_dataset,
    get_dataset_stats,
    load_dataset,
    sample_questions,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _raw_entry(
    _id: str = "abc123",
    question: str = "Were A and B of the same nationality?",
    answer: str = "yes",
    type_: str = "comparison",
    level: str = "easy",
) -> dict:
    """Build a minimal valid raw HotpotQA JSON entry."""
    return {
        "_id": _id,
        "question": question,
        "answer": answer,
        "type": type_,
        "level": level,
        "supporting_facts": [["Title A", 0], ["Title B", 1]],
        "context": [
            ["Title A", ["Sentence one about A."]],
            ["Title B", ["Sentence one about B.", "Sentence two about B."]],
        ],
    }


def _make_question(
    id: str = "q1",
    question: str = "What is X?",
    answer: str = "X",
    qtype: str = "comparison",
    level: str = "easy",
) -> HotpotQuestion:
    """Construct a HotpotQuestion directly (bypasses load_dataset)."""
    return HotpotQuestion(
        id=id,
        question=question,
        answer=answer,
        type=qtype,
        level=level,
        supporting_facts=(("Title A", 0),),
        context_titles=("Title A", "Title B"),
    )


def _write_fixture(tmp_path, entries: list[dict]):
    """Write a list of raw entries as a JSON file and return its path."""
    fixture = tmp_path / "hotpot_fixture.json"
    fixture.write_text(json.dumps(entries), encoding="utf-8")
    return fixture


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_load_dataset_parses_valid_entries(tmp_path):
    """load_dataset returns one HotpotQuestion per valid entry, with correct fields."""
    entries = [
        _raw_entry(_id="id1", question="Q1?", answer="A1", type_="comparison", level="easy"),
        _raw_entry(_id="id2", question="Q2?", answer="A2", type_="bridge", level="medium"),
        _raw_entry(_id="id3", question="Q3?", answer="A3", type_="comparison", level="hard"),
    ]
    fixture = _write_fixture(tmp_path, entries)

    questions = load_dataset(path=fixture)

    assert len(questions) == 3
    assert questions[0].id == "id1"
    assert questions[0].question == "Q1?"
    assert questions[0].answer == "A1"
    assert questions[0].type == "comparison"
    assert questions[0].level == "easy"
    assert questions[0].supporting_facts == (("Title A", 0), ("Title B", 1))
    assert questions[0].context_titles == ("Title A", "Title B")

    assert questions[1].type == "bridge"
    assert questions[1].level == "medium"
    assert questions[2].level == "hard"


def test_load_dataset_skips_malformed(tmp_path, caplog):
    """load_dataset skips entries missing required fields or with invalid type values."""
    missing_answer = _raw_entry(_id="bad1")
    del missing_answer["answer"]

    bad_type = _raw_entry(_id="bad2")
    bad_type["type"] = "foo"

    entries = [
        _raw_entry(_id="good1"),
        _raw_entry(_id="good2"),
        missing_answer,
        bad_type,
    ]
    fixture = _write_fixture(tmp_path, entries)

    with caplog.at_level(logging.INFO, logger="eval.hotpot.loader"):
        questions = load_dataset(path=fixture)

    assert len(questions) == 2
    assert {q.id for q in questions} == {"good1", "good2"}
    assert any("skipped 2 malformed" in r.message for r in caplog.records)


def test_sample_questions_is_deterministic_with_seed(tmp_path):
    """sample_questions returns the same questions on repeated calls with the same seed."""
    entries = [_raw_entry(_id=f"q{i}") for i in range(10)]
    fixture = _write_fixture(tmp_path, entries)
    questions = load_dataset(path=fixture)

    sample_a = sample_questions(questions, n=3, seed=42)
    sample_b = sample_questions(questions, n=3, seed=42)

    assert sample_a == sample_b
    assert len(sample_a) == 3


def test_sample_questions_respects_type_filter(tmp_path):
    """sample_questions returns only questions matching type_filter."""
    entries = [
        _raw_entry(_id="c1", type_="comparison"),
        _raw_entry(_id="c2", type_="comparison"),
        _raw_entry(_id="c3", type_="comparison"),
        _raw_entry(_id="b1", type_="bridge"),
        _raw_entry(_id="b2", type_="bridge"),
    ]
    fixture = _write_fixture(tmp_path, entries)
    questions = load_dataset(path=fixture)

    sample = sample_questions(questions, n=2, type_filter="comparison", seed=0)

    assert len(sample) == 2
    assert all(q.type == "comparison" for q in sample)


def test_sample_questions_raises_when_n_too_large(tmp_path):
    """sample_questions raises ValueError when n exceeds the filtered pool size."""
    entries = [_raw_entry(_id=f"q{i}") for i in range(5)]
    fixture = _write_fixture(tmp_path, entries)
    questions = load_dataset(path=fixture)

    with pytest.raises(ValueError, match="Requested 10 questions but only 5"):
        sample_questions(questions, n=10)


def test_get_dataset_stats_counts_by_type_and_level(tmp_path):
    """get_dataset_stats returns correct counts broken down by type and level."""
    entries = [
        _raw_entry(_id="1", type_="comparison", level="easy"),
        _raw_entry(_id="2", type_="comparison", level="hard"),
        _raw_entry(_id="3", type_="bridge", level="easy"),
        _raw_entry(_id="4", type_="bridge", level="medium"),
    ]
    fixture = _write_fixture(tmp_path, entries)
    questions = load_dataset(path=fixture)

    stats = get_dataset_stats(questions)

    assert stats["total"] == 4
    assert stats["by_type"]["comparison"] == 2
    assert stats["by_type"]["bridge"] == 2
    assert stats["by_level"]["easy"] == 2
    assert stats["by_level"]["medium"] == 1
    assert stats["by_level"]["hard"] == 1
    assert stats["by_type_and_level"][("comparison", "easy")] == 1
    assert stats["by_type_and_level"][("comparison", "hard")] == 1
    assert stats["by_type_and_level"][("bridge", "easy")] == 1
    assert stats["by_type_and_level"][("bridge", "medium")] == 1


def test_download_dataset_skips_when_cached(tmp_path, monkeypatch):
    """download_dataset returns immediately if the cache file exists, without downloading."""
    fake_cache = tmp_path / "hotpot_dev_distractor_v1.json"
    fake_cache.write_text("[]", encoding="utf-8")

    monkeypatch.setattr("eval.hotpot.loader.CACHE_FILE", fake_cache)

    def _should_not_be_called(*args, **kwargs):
        raise AssertionError("urlretrieve must not be called when cache exists")

    monkeypatch.setattr("eval.hotpot.loader.urlretrieve", _should_not_be_called)

    result = download_dataset()

    assert result == fake_cache
