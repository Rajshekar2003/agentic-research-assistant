"""HotpotQA eval runner — sequential calls to both endpoints, JSONL output, resume support.

Usage (from backend/):
    python -m eval.hotpot.runner --n 5
    python -m eval.hotpot.runner --n 100 --seed 7 --type bridge --output eval/hotpot/runs/my-run.jsonl

Output format: one JSON object per line (JSONL). Each line is a QuestionResult, fully
self-contained. The scorer (Day 17) reads this file with line-by-line json.loads.
"""

import argparse
import asyncio
import dataclasses
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import httpx

from eval.hotpot.loader import HotpotQuestion, load_dataset, sample_questions

logger = logging.getLogger(__name__)

TIMEOUT_S = 90.0

_TRANSIENT_HTTP_STATUSES = frozenset({429, 503, 504})


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class EndpointResult:
    """Result of a single POST request to one endpoint.

    Attributes:
        status: Outcome category — "ok", "http_error", "timeout", or "exception".
        http_status: HTTP status code, or None if the request never completed.
        elapsed_ms: Client-side wall-clock time in milliseconds (always populated).
        answer: Extracted answer string from a 200 response, else None.
        sources: List of {title, url} dicts from a 200 response, else None.
        facts: Graph-only — list of {claim, sources} dicts, else None.
        critic_verdict: Graph-only — "approve" or "revise" from the Critic node, else None.
        revisions: Graph-only — number of revision passes taken, else None.
        error: Human-readable error description when status != "ok", else None.
    """

    status: Literal["ok", "http_error", "timeout", "exception"]
    http_status: int | None
    elapsed_ms: int | None
    answer: str | None
    sources: list[dict] | None
    facts: list[dict] | None
    critic_verdict: str | None
    revisions: int | None
    error: str | None


@dataclasses.dataclass
class QuestionResult:
    """Combined result for one HotpotQA question against both endpoints.

    Attributes:
        question_id: HotpotQA question ID (_id field).
        question_text: The natural-language question string.
        ground_truth_answer: Gold answer string from the dataset.
        question_type: "comparison" or "bridge".
        question_level: "easy", "medium", or "hard".
        baseline: EndpointResult from POST /research.
        graph: EndpointResult from POST /research/graph.
        timestamp: ISO-8601 UTC timestamp when this result was recorded.
    """

    question_id: str
    question_text: str
    ground_truth_answer: str
    question_type: str
    question_level: str
    baseline: EndpointResult
    graph: EndpointResult
    timestamp: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _is_transient(result: EndpointResult) -> bool:
    """Return True if result represents a transient failure worth retrying.

    Transient: HTTP 429/503/504, client-side timeouts, and network-level exceptions
    (connection resets, DNS failures, etc.). Hard failures (4xx other than 429) are
    not retried.
    """
    if result.status == "timeout":
        return True
    if result.status == "exception":
        return True
    if result.status == "http_error" and result.http_status in _TRANSIENT_HTTP_STATUSES:
        return True
    return False


async def _call_endpoint(
    client: httpx.AsyncClient,
    url: str,
    query: str,
) -> EndpointResult:
    """POST query to one endpoint and return a structured EndpointResult.

    Args:
        client: Shared httpx AsyncClient.
        url: Full URL including path (e.g. http://localhost:8000/research).
        query: The natural-language query string to send.

    Returns:
        EndpointResult — never raises; all failure modes are captured as status fields.
    """
    t0 = time.perf_counter()
    try:
        resp = await client.post(url, json={"query": query})
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        if resp.status_code == 200:
            data = resp.json()
            return EndpointResult(
                status="ok",
                http_status=resp.status_code,
                elapsed_ms=elapsed_ms,
                answer=data.get("answer"),
                sources=data.get("sources"),
                facts=data.get("facts"),
                critic_verdict=data.get("critic_verdict"),
                revisions=data.get("revisions"),
                error=None,
            )

        return EndpointResult(
            status="http_error",
            http_status=resp.status_code,
            elapsed_ms=elapsed_ms,
            answer=None,
            sources=None,
            facts=None,
            critic_verdict=None,
            revisions=None,
            error=resp.text[:500],
        )

    except (asyncio.TimeoutError, httpx.TimeoutException):
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return EndpointResult(
            status="timeout",
            http_status=None,
            elapsed_ms=elapsed_ms,
            answer=None,
            sources=None,
            facts=None,
            critic_verdict=None,
            revisions=None,
            error="httpx client timeout",
        )

    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return EndpointResult(
            status="exception",
            http_status=None,
            elapsed_ms=elapsed_ms,
            answer=None,
            sources=None,
            facts=None,
            critic_verdict=None,
            revisions=None,
            error=str(exc)[:500],
        )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def run_single_question(
    client: httpx.AsyncClient,
    question: HotpotQuestion,
    base_url: str,
    max_retries: int = 3,
    _retry_counter: list[int] | None = None,
) -> QuestionResult:
    """Call both endpoints for one question and return a combined QuestionResult.

    Calls are sequential: baseline first, then graph. Each endpoint is retried up to
    max_retries times on transient failures (HTTP 429/503/504, timeouts, network
    exceptions) with exponential backoff (2**attempt seconds, capped at 30s).
    Hard failures (4xx other than 429) are returned immediately without retrying.

    Args:
        client: Shared httpx AsyncClient with timeout already configured.
        question: The HotpotQA question to evaluate.
        base_url: Base URL of the research assistant API (no trailing slash).
        max_retries: Number of retry attempts per endpoint on transient failures.
            Default 3. Set to 0 to disable retrying.
        _retry_counter: Optional mutable [int] list incremented once per retry attempt
            across both endpoints. Pass this from run_eval to aggregate counts.

    Returns:
        QuestionResult with EndpointResult for both baseline and graph endpoints.
    """

    async def _call_with_retry(url: str, endpoint_name: str) -> EndpointResult:
        for attempt in range(max_retries + 1):
            result = await _call_endpoint(client, url, question.question)

            if result.status == "ok":
                return result

            if not _is_transient(result):
                return result  # hard failure — return immediately, no retry

            if attempt < max_retries:
                if _retry_counter is not None:
                    _retry_counter[0] += 1
                error_summary = result.error or str(result.http_status)
                logger.warning(
                    "retry attempt=%d/%d for %s on %s: %s",
                    attempt + 1,
                    max_retries,
                    question.id,
                    endpoint_name,
                    error_summary,
                )
                sleep_s = min(2**attempt, 30)
                await asyncio.sleep(sleep_s)
            else:
                # Exhausted all retries on a transient failure.
                if max_retries > 0:
                    error_summary = result.error or str(result.http_status)
                    return EndpointResult(
                        status="http_error",
                        http_status=result.http_status,
                        elapsed_ms=result.elapsed_ms,
                        answer=None,
                        sources=None,
                        facts=None,
                        critic_verdict=None,
                        revisions=None,
                        error=f"max retries exceeded: {error_summary}",
                    )
                return result  # max_retries=0: just return the single attempt result

        raise RuntimeError("unreachable")

    baseline = await _call_with_retry(f"{base_url}/research", "baseline")
    graph = await _call_with_retry(f"{base_url}/research/graph", "graph")
    return QuestionResult(
        question_id=question.id,
        question_text=question.question,
        ground_truth_answer=question.answer,
        question_type=question.type,
        question_level=question.level,
        baseline=baseline,
        graph=graph,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


async def run_eval(
    questions: list[HotpotQuestion],
    base_url: str,
    output_path: Path,
    delay_seconds: float = 3.0,
    resume: bool = True,
    max_retries: int = 3,
    _transport: httpx.AsyncBaseTransport | None = None,
) -> dict:
    """Run the evaluation sequentially over all questions, writing JSONL after each.

    Questions are processed one at a time (no concurrency) to respect Tavily and
    Groq rate limits. Each result is written, flushed, and fsync'd immediately after
    the pair of endpoint calls completes, so a crash mid-run leaves a valid,
    partially-complete output file that can be resumed on the next invocation.

    Args:
        questions: Pre-sampled list of HotpotQuestion instances to evaluate.
        base_url: Base URL of the running research assistant server.
        output_path: Path to the JSONL output file. Created if absent; appended if
            resume=True and it already exists.
        delay_seconds: Seconds to sleep between questions. Default 3.0 — Groq-rate-limit-
            friendly; smaller values may work at small N but will hit 429s at scale.
        resume: If True and output_path exists, skip questions whose IDs are already
            present in the file. Default True.
        max_retries: Number of retry attempts per endpoint on transient failures. Default 3.
        _transport: Internal — inject a custom httpx transport (e.g. MockTransport
            in tests). Production callers should leave this as None.

    Returns:
        dict with keys: both_ok, baseline_errors, graph_errors, total_retries, total_ran.
    """
    completed_ids: set[str] = set()
    if resume and output_path.exists():
        with output_path.open(encoding="utf-8") as fh:
            for raw_line in fh:
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    data = json.loads(stripped)
                    completed_ids.add(data["question_id"])
                except (json.JSONDecodeError, KeyError):
                    logger.warning("Skipping malformed resume line: %r", stripped[:80])

    to_run = [q for q in questions if q.id not in completed_ids]
    n_total = len(questions)
    n_done = n_total - len(to_run)
    n_todo = len(to_run)

    print(
        f"Running {n_total} questions ({n_done} already complete, {n_todo} to do). "
        f"Output: {output_path}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    baseline_errors = 0
    graph_errors = 0
    both_ok = 0
    retry_counter = [0]

    async with httpx.AsyncClient(timeout=TIMEOUT_S, transport=_transport) as client:
        for idx, question in enumerate(to_run):
            i = n_done + idx + 1
            print(f"[{i}/{n_total}] {question.id}: {question.question[:80]}...")

            result = await run_single_question(
                client,
                question,
                base_url,
                max_retries=max_retries,
                _retry_counter=retry_counter,
            )

            with output_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(dataclasses.asdict(result)) + "\n")
                fh.flush()
                # Belt-and-braces durability: flush + fsync per line. Adds ~5ms overhead
                # per question but eliminates entire classes of crash-related data loss.
                os.fsync(fh.fileno())

            if result.baseline.status != "ok":
                baseline_errors += 1
            if result.graph.status != "ok":
                graph_errors += 1
            if result.baseline.status == "ok" and result.graph.status == "ok":
                both_ok += 1

            if idx < n_todo - 1:
                await asyncio.sleep(delay_seconds)

    total_retries = retry_counter[0]
    print(
        f"Eval complete. {both_ok}/{n_todo} both ok, "
        f"{baseline_errors} baseline errors, {graph_errors} graph errors, "
        f"{total_retries} retries across all questions. "
        f"Output: {output_path}"
    )

    return {
        "both_ok": both_ok,
        "baseline_errors": baseline_errors,
        "graph_errors": graph_errors,
        "total_retries": total_retries,
        "total_ran": n_todo,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="HotpotQA eval runner — sequential baseline + graph calls, JSONL output."
    )
    parser.add_argument(
        "--n",
        type=int,
        default=5,
        metavar="N",
        help="Number of questions to sample (default: 5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="N",
        help="Random seed for deterministic sampling (default: 42)",
    )
    parser.add_argument(
        "--type",
        choices=["comparison", "bridge", "both"],
        default="both",
        dest="qtype",
        help="Question type filter (default: both)",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        metavar="PATH",
        help="Output JSONL path (default: eval/hotpot/runs/<ISO timestamp>.jsonl)",
    )
    parser.add_argument(
        "--delay-seconds",
        type=float,
        default=3.0,
        metavar="N",
        help=(
            "Seconds to sleep between questions (default: 3.0 — Groq-rate-limit-friendly; "
            "smaller values may work at small N but will hit 429s at scale)"
        ),
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume — re-run all questions even if output file exists",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        metavar="N",
        help=(
            "Max retries per endpoint on transient errors (429, 503, 504, timeouts). "
            "Uses exponential backoff capped at 30s. Default: 3."
        ),
    )
    parser.add_argument(
        "--target",
        type=int,
        default=None,
        metavar="N",
        help=(
            "Target number of successful (both-ok) results. If higher than --n, the runner "
            "samples additional questions until the target is reached or the dataset is "
            "exhausted. Default: same as --n."
        ),
    )
    args = parser.parse_args()

    target = args.target if args.target is not None else args.n

    if args.output is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = Path(__file__).parent / "runs" / f"{ts}.jsonl"

    type_filter = None if args.qtype == "both" else args.qtype

    all_questions = load_dataset()

    # Pre-sample a pool large enough to cover --target with buffer for failures.
    # Using the same seed guarantees that full_pool[:n] == the original --n sample,
    # so resume behaviour is stable when --target triggers extra questions.
    if target > args.n:
        pool_n = min(target * 2, len(all_questions))
        full_pool = sample_questions(all_questions, n=pool_n, seed=args.seed, type_filter=type_filter)
        initial_sample = full_pool[: args.n]
        reserve = full_pool[args.n :]
    else:
        initial_sample = sample_questions(
            all_questions, n=args.n, seed=args.seed, type_filter=type_filter
        )
        reserve = []

    stats = asyncio.run(
        run_eval(
            questions=initial_sample,
            base_url=args.base_url,
            output_path=args.output,
            delay_seconds=args.delay_seconds,
            resume=not args.no_resume,
            max_retries=args.max_retries,
        )
    )

    total_successful = stats["both_ok"]
    total_ran = stats["total_ran"]

    # --target top-up loop: run reserve questions until target is reached.
    if target > args.n and total_successful < target and reserve:
        remaining = list(reserve)
        while total_successful < target and remaining:
            needed = target - total_successful
            batch = remaining[: needed + 5]  # small buffer per batch
            remaining = remaining[len(batch) :]

            batch_stats = asyncio.run(
                run_eval(
                    questions=batch,
                    base_url=args.base_url,
                    output_path=args.output,
                    delay_seconds=args.delay_seconds,
                    resume=not args.no_resume,
                    max_retries=args.max_retries,
                )
            )
            total_successful += batch_stats["both_ok"]
            total_ran += batch_stats["total_ran"]

    if args.target is not None:
        if total_successful >= target:
            print(f"Target {target} reached")
        else:
            print(
                f"Target {target} not reached; ran {total_ran} questions, "
                f"{total_successful} successful"
            )
