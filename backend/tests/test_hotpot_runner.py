"""Tests for the HotpotQA eval runner (eval/hotpot/runner.py).

All tests use httpx.MockTransport — no real network calls are made.
"""

import json

import httpx
import pytest

from eval.hotpot.loader import HotpotQuestion
from eval.hotpot.runner import EndpointResult, QuestionResult, run_eval, run_single_question


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _make_question(
    id: str = "q1",
    question: str = "What is the capital of France?",
    answer: str = "Paris",
    qtype: str = "comparison",
    level: str = "easy",
) -> HotpotQuestion:
    return HotpotQuestion(
        id=id,
        question=question,
        answer=answer,
        type=qtype,
        level=level,
        supporting_facts=(("Title A", 0),),
        context_titles=("Title A",),
    )


_BASELINE_JSON = {
    "answer": "Paris",
    "sources": [{"title": "Wikipedia", "url": "https://en.wikipedia.org/wiki/Paris"}],
}

_GRAPH_JSON = {
    "answer": "Paris is the capital of France.",
    "sources": [{"title": "Wikipedia", "url": "https://en.wikipedia.org/wiki/Paris"}],
    "facts": [{"claim": "Paris is the capital", "sources": [0]}],
    "critic_verdict": "approve",
    "revisions": 0,
}


def _route(request: httpx.Request, *, baseline: httpx.Response, graph: httpx.Response) -> httpx.Response:
    """Route a mock request to the right response based on URL path."""
    url = str(request.url)
    if "/research/graph" in url:
        return graph
    if "/research" in url:
        return baseline
    return httpx.Response(404, text="Not found")


# ---------------------------------------------------------------------------
# run_single_question tests
# ---------------------------------------------------------------------------


async def test_run_single_question_happy_path():
    """Both endpoints return 200 — EndpointResults have status='ok' and correct fields."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _route(
            request,
            baseline=httpx.Response(200, json=_BASELINE_JSON),
            graph=httpx.Response(200, json=_GRAPH_JSON),
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        result = await run_single_question(client, _make_question(), "http://localhost:8000")

    assert isinstance(result, QuestionResult)
    assert result.question_id == "q1"
    assert result.ground_truth_answer == "Paris"

    assert result.baseline.status == "ok"
    assert result.baseline.http_status == 200
    assert result.baseline.answer == "Paris"
    assert result.baseline.sources == _BASELINE_JSON["sources"]
    assert result.baseline.elapsed_ms is not None
    assert result.baseline.error is None

    assert result.graph.status == "ok"
    assert result.graph.http_status == 200
    assert result.graph.answer == "Paris is the capital of France."
    assert result.graph.critic_verdict == "approve"
    assert result.graph.revisions == 0
    assert result.graph.error is None


async def test_run_single_question_baseline_503():
    """Baseline 503 → status='http_error'; graph 200 → status='ok'. Does not raise."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _route(
            request,
            baseline=httpx.Response(503, text="Service Unavailable"),
            graph=httpx.Response(200, json=_GRAPH_JSON),
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        # max_retries=0: verify the single-attempt error path without retry overhead.
        result = await run_single_question(
            client, _make_question(), "http://localhost:8000", max_retries=0
        )

    assert result.baseline.status == "http_error"
    assert result.baseline.http_status == 503
    assert "Service Unavailable" in result.baseline.error
    assert result.baseline.answer is None

    assert result.graph.status == "ok"
    assert result.graph.answer == _GRAPH_JSON["answer"]


async def test_run_single_question_graph_504():
    """Baseline 200 → status='ok'; graph 504 → status='http_error' with http_status=504."""

    def handler(request: httpx.Request) -> httpx.Response:
        return _route(
            request,
            baseline=httpx.Response(200, json=_BASELINE_JSON),
            graph=httpx.Response(504, text="Gateway Timeout"),
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        # max_retries=0: verify the single-attempt error path without retry overhead.
        result = await run_single_question(
            client, _make_question(), "http://localhost:8000", max_retries=0
        )

    assert result.baseline.status == "ok"
    assert result.baseline.answer == "Paris"

    assert result.graph.status == "http_error"
    assert result.graph.http_status == 504
    assert result.graph.answer is None
    assert result.graph.error is not None


async def test_run_single_question_network_exception():
    """Handler raises NetworkError → both endpoints get status='exception' with original message."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.NetworkError("connection refused")

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        # max_retries=0: verify the single-attempt exception path without retry overhead.
        result = await run_single_question(
            client, _make_question(), "http://localhost:8000", max_retries=0
        )

    assert result.baseline.status == "exception"
    assert "connection refused" in result.baseline.error
    assert result.baseline.http_status is None

    assert result.graph.status == "exception"
    assert "connection refused" in result.graph.error


async def test_run_single_question_captures_graph_facts():
    """Graph response with facts/critic_verdict/revisions → fields populated in EndpointResult."""
    graph_with_facts = {
        "answer": "Multi-hop answer.",
        "sources": [],
        "facts": [
            {"claim": "Claim A", "sources": [0]},
            {"claim": "Claim B", "sources": [1]},
        ],
        "critic_verdict": "revise",
        "revisions": 1,
    }

    def handler(request: httpx.Request) -> httpx.Response:
        return _route(
            request,
            baseline=httpx.Response(200, json=_BASELINE_JSON),
            graph=httpx.Response(200, json=graph_with_facts),
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        result = await run_single_question(client, _make_question(), "http://localhost:8000")

    assert result.graph.status == "ok"
    assert result.graph.facts == graph_with_facts["facts"]
    assert result.graph.critic_verdict == "revise"
    assert result.graph.revisions == 1
    assert len(result.graph.facts) == 2


# ---------------------------------------------------------------------------
# run_single_question retry tests
# ---------------------------------------------------------------------------


async def test_run_single_question_retries_on_429(monkeypatch):
    """Handler returns 429 twice then 200 — baseline succeeds after 2 retries, counter=2."""
    monkeypatch.setattr("eval.hotpot.runner.asyncio.sleep", lambda _: _noop())

    call_count = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        call_count[0] += 1
        if call_count[0] <= 2:
            return httpx.Response(429, text="Too Many Requests")
        return _route(
            request,
            baseline=httpx.Response(200, json=_BASELINE_JSON),
            graph=httpx.Response(200, json=_GRAPH_JSON),
        )

    transport = httpx.MockTransport(handler)
    counter = [0]
    async with httpx.AsyncClient(transport=transport) as client:
        result = await run_single_question(
            client, _make_question(), "http://localhost:8000",
            max_retries=3, _retry_counter=counter,
        )

    assert result.baseline.status == "ok"
    assert result.baseline.answer == "Paris"
    assert counter[0] == 2


async def test_run_single_question_retries_on_503(monkeypatch):
    """Handler returns 503 twice then 200 — baseline succeeds after 2 retries, counter=2."""
    monkeypatch.setattr("eval.hotpot.runner.asyncio.sleep", lambda _: _noop())

    call_count = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        call_count[0] += 1
        if call_count[0] <= 2:
            return httpx.Response(503, text="Service Unavailable")
        return _route(
            request,
            baseline=httpx.Response(200, json=_BASELINE_JSON),
            graph=httpx.Response(200, json=_GRAPH_JSON),
        )

    transport = httpx.MockTransport(handler)
    counter = [0]
    async with httpx.AsyncClient(transport=transport) as client:
        result = await run_single_question(
            client, _make_question(), "http://localhost:8000",
            max_retries=3, _retry_counter=counter,
        )

    assert result.baseline.status == "ok"
    assert counter[0] == 2


async def test_run_single_question_retries_on_timeout(monkeypatch):
    """Handler raises TimeoutException twice then returns 200 — succeeds after 2 retries."""
    monkeypatch.setattr("eval.hotpot.runner.asyncio.sleep", lambda _: _noop())

    call_count = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        call_count[0] += 1
        if call_count[0] <= 2:
            raise httpx.TimeoutException("read timeout")
        return _route(
            request,
            baseline=httpx.Response(200, json=_BASELINE_JSON),
            graph=httpx.Response(200, json=_GRAPH_JSON),
        )

    transport = httpx.MockTransport(handler)
    counter = [0]
    async with httpx.AsyncClient(transport=transport) as client:
        result = await run_single_question(
            client, _make_question(), "http://localhost:8000",
            max_retries=3, _retry_counter=counter,
        )

    assert result.baseline.status == "ok"
    assert counter[0] == 2


async def test_run_single_question_does_not_retry_on_400():
    """Handler returns 400 — hard failure, no retry, counter stays 0."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, text="Bad Request")

    transport = httpx.MockTransport(handler)
    counter = [0]
    async with httpx.AsyncClient(transport=transport) as client:
        result = await run_single_question(
            client, _make_question(), "http://localhost:8000",
            max_retries=3, _retry_counter=counter,
        )

    assert result.baseline.status == "http_error"
    assert result.baseline.http_status == 400
    assert counter[0] == 0


async def test_run_single_question_gives_up_after_max_retries(monkeypatch):
    """Handler always returns 429; after max_retries=3 exhausted, baseline is 'max retries exceeded'."""
    monkeypatch.setattr("eval.hotpot.runner.asyncio.sleep", lambda _: _noop())

    call_count = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        call_count[0] += 1
        # First 4 calls (all baseline attempts) return 429; graph then gets 200.
        if call_count[0] <= 4:
            return httpx.Response(429, text="Too Many Requests")
        return _route(
            request,
            baseline=httpx.Response(200, json=_BASELINE_JSON),
            graph=httpx.Response(200, json=_GRAPH_JSON),
        )

    transport = httpx.MockTransport(handler)
    counter = [0]
    async with httpx.AsyncClient(transport=transport) as client:
        result = await run_single_question(
            client, _make_question(), "http://localhost:8000",
            max_retries=3, _retry_counter=counter,
        )

    assert result.baseline.status == "http_error"
    assert result.baseline.error is not None
    assert "max retries exceeded" in result.baseline.error
    assert counter[0] == 3


# ---------------------------------------------------------------------------
# run_eval tests
# ---------------------------------------------------------------------------


async def test_run_eval_writes_jsonl(tmp_path):
    """3 questions → output JSONL has 3 lines, each parseable with all required fields."""
    questions = [_make_question(id=f"q{i}", question=f"Question {i}?") for i in range(3)]
    output = tmp_path / "out.jsonl"

    def handler(request: httpx.Request) -> httpx.Response:
        return _route(
            request,
            baseline=httpx.Response(200, json=_BASELINE_JSON),
            graph=httpx.Response(200, json=_GRAPH_JSON),
        )

    transport = httpx.MockTransport(handler)
    await run_eval(
        questions,
        "http://localhost:8000",
        output,
        delay_seconds=0,
        _transport=transport,
    )

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3

    required_fields = {
        "question_id",
        "question_text",
        "ground_truth_answer",
        "question_type",
        "question_level",
        "baseline",
        "graph",
        "timestamp",
    }
    for line in lines:
        data = json.loads(line)
        assert required_fields <= data.keys(), f"Missing fields in: {data.keys()}"
        assert data["baseline"]["status"] == "ok"
        assert data["graph"]["status"] == "ok"


async def test_run_eval_resumes_from_existing(tmp_path):
    """Pre-existing file with 2 IDs → only 1 new line added; old lines unchanged."""
    q0 = _make_question(id="q0", question="Q0?")
    q1 = _make_question(id="q1", question="Q1?")
    q2 = _make_question(id="q2", question="Q2?")

    output = tmp_path / "out.jsonl"
    existing_q0 = {"question_id": "q0", "marker": "original"}
    existing_q1 = {"question_id": "q1", "marker": "original"}
    with output.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps(existing_q0) + "\n")
        fh.write(json.dumps(existing_q1) + "\n")

    def handler(request: httpx.Request) -> httpx.Response:
        return _route(
            request,
            baseline=httpx.Response(200, json=_BASELINE_JSON),
            graph=httpx.Response(200, json=_GRAPH_JSON),
        )

    transport = httpx.MockTransport(handler)
    await run_eval(
        [q0, q1, q2],
        "http://localhost:8000",
        output,
        delay_seconds=0,
        _transport=transport,
    )

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3, f"Expected 3 lines, got {len(lines)}"

    # Original lines must be byte-for-byte unchanged.
    assert json.loads(lines[0]) == existing_q0
    assert json.loads(lines[1]) == existing_q1

    # Third line is the new result for q2 only.
    new_result = json.loads(lines[2])
    assert new_result["question_id"] == "q2"
    assert "marker" not in new_result  # Not a copy of the pre-populated stubs.


async def test_run_eval_continues_on_error(tmp_path):
    """When 2nd question's endpoints fail, all 3 lines appear; no exception bubbles up."""
    questions = [
        _make_question(id="q0", question="Q0?"),
        _make_question(id="q1", question="Q1?"),
        _make_question(id="q2", question="Q2?"),
    ]
    output = tmp_path / "out.jsonl"

    call_count = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        call_count[0] += 1
        # Calls 3 and 4 correspond to q1 (baseline + graph). Force them to fail.
        if call_count[0] in (3, 4):
            raise httpx.NetworkError("forced failure for q1")
        return _route(
            request,
            baseline=httpx.Response(200, json=_BASELINE_JSON),
            graph=httpx.Response(200, json=_GRAPH_JSON),
        )

    transport = httpx.MockTransport(handler)
    # max_retries=0: call_count positions are deterministic only without retries.
    await run_eval(
        questions,
        "http://localhost:8000",
        output,
        delay_seconds=0,
        max_retries=0,
        _transport=transport,
    )

    lines = output.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3

    results = [json.loads(line) for line in lines]
    ids = [r["question_id"] for r in results]
    assert ids == ["q0", "q1", "q2"]

    assert results[0]["baseline"]["status"] == "ok"
    assert results[0]["graph"]["status"] == "ok"

    assert results[1]["baseline"]["status"] == "exception"
    assert results[1]["graph"]["status"] == "exception"

    assert results[2]["baseline"]["status"] == "ok"
    assert results[2]["graph"]["status"] == "ok"


async def test_run_eval_aggregates_retry_counts(monkeypatch, tmp_path):
    """3 questions, graph retries once each → stats reports total_retries=3."""
    monkeypatch.setattr("eval.hotpot.runner.asyncio.sleep", lambda _: _noop())

    graph_call_count = [0]

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/research/graph" in url:
            graph_call_count[0] += 1
            # Odd-numbered graph calls fail (first attempt per question); even ones succeed.
            if graph_call_count[0] % 2 == 1:
                return httpx.Response(429, text="Rate limited")
        return _route(
            request,
            baseline=httpx.Response(200, json=_BASELINE_JSON),
            graph=httpx.Response(200, json=_GRAPH_JSON),
        )

    questions = [_make_question(id=f"q{i}", question=f"Q{i}?") for i in range(3)]
    output = tmp_path / "out.jsonl"

    transport = httpx.MockTransport(handler)
    stats = await run_eval(
        questions,
        "http://localhost:8000",
        output,
        delay_seconds=0,
        max_retries=3,
        _transport=transport,
    )

    assert stats["total_retries"] == 3
    assert stats["both_ok"] == 3


# ---------------------------------------------------------------------------
# Async no-op helper (used to replace asyncio.sleep in monkeypatched tests)
# ---------------------------------------------------------------------------


async def _noop():
    pass
