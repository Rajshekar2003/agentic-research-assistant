"""Smoke eval runner — hits both /research endpoints with curated questions, writes a markdown report.

Usage (from backend/):
    python -m eval.run_eval
    python -m eval.run_eval --questions eval/questions.yaml --output eval/reports/my-run.md
"""

import argparse
import asyncio
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import yaml

BASE_URL = "http://localhost:8000"
TIMEOUT_S = 90.0
DEFAULT_QUESTIONS = Path(__file__).parent / "questions.yaml"
DEFAULT_REPORTS = Path(__file__).parent / "reports"


def load_questions(path: Path) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["questions"]


async def call_endpoint(client: httpx.AsyncClient, path: str, query: str) -> dict:
    """POST to one endpoint and return a normalized result dict.

    On success: full JSON response dict with elapsed_ms ensured.
    On HTTP error or exception: dict with '_error' key describing the failure.
    """
    t0 = time.perf_counter()
    try:
        resp = await client.post(f"{BASE_URL}{path}", json={"query": query})
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        if resp.status_code == 200:
            data = resp.json()
            data.setdefault("elapsed_ms", elapsed_ms)
            return data
        return {
            "_error": f"HTTP {resp.status_code}: {resp.text[:300]}",
            "elapsed_ms": elapsed_ms,
        }
    except Exception as exc:
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return {"_error": f"{type(exc).__name__}: {exc}", "elapsed_ms": elapsed_ms}


# ---------------------------------------------------------------------------
# Markdown rendering helpers
# ---------------------------------------------------------------------------


def _source_list(sources: list[dict]) -> str:
    if not sources:
        return "_No sources._"
    return "\n".join(
        f"- [{i}] {s.get('title', 'Untitled')} — {s.get('url', '')}"
        for i, s in enumerate(sources, 1)
    )


def _fact_list(facts: list[dict]) -> str:
    lines = ["*Verified facts:*"]
    for f in facts:
        src_ids = ", ".join(str(x) for x in f.get("sources", []))
        lines.append(f'- "{f["claim"]}" (sources: [{src_ids}])')
    return "\n".join(lines)


def _render_question(q: dict, baseline: dict, graph: dict) -> str:
    qid, cat, text = q["id"], q["category"], q["text"]
    notes = q.get("notes", "")

    parts: list[str] = [
        f"## {qid} — {cat}",
        "",
        f"**Question:** {text}",
        "",
    ]
    if notes:
        parts += [f"**Reviewer notes:** {notes}", ""]

    # Baseline block
    b_ms = baseline.get("elapsed_ms", 0)
    if "_error" in baseline:
        parts += [f"### Baseline ({b_ms}ms) — ERROR", "", f"_{baseline['_error']}_", ""]
    else:
        parts += [
            f"### Baseline ({b_ms}ms)",
            "",
            baseline.get("answer", "_No answer._"),
            "",
            "*Sources:*",
            _source_list(baseline.get("sources", [])),
            "",
        ]

    # Graph block
    g_ms = graph.get("elapsed_ms", 0)
    if "_error" in graph:
        parts += [f"### Graph ({g_ms}ms) — ERROR", "", f"_{graph['_error']}_", ""]
    else:
        facts = graph.get("facts") or []
        meta = (
            f"facts_count={len(facts)}, "
            f"critic_verdict={graph.get('critic_verdict', 'n/a')}, "
            f"revisions={graph.get('revisions', 0)}"
        )
        parts += [
            f"### Graph ({g_ms}ms, {meta})",
            "",
            graph.get("answer", "_No answer._"),
            "",
            "*Sources:*",
            _source_list(graph.get("sources", [])),
        ]
        if facts:
            parts += ["", _fact_list(facts)]
        parts.append("")

    # Scoring table (fill in by hand)
    parts += [
        "### Scoring (fill in by hand)",
        "",
        "| Criterion | Baseline (0-2) | Graph (0-2) | Notes |",
        "|-----------|----------------|-------------|-------|",
        "| Groundedness — every claim supported by cited sources |  |  |  |",
        "| Citation accuracy — citations point to correct sources |  |  |  |",
        "| Completeness — addresses all parts of the question |  |  |  |",
        "| Hallucination check — 0 if any fabricated claims, 2 if none |  |  |  |",
        "| **Total (0-8)** |  |  |  |",
        "",
        "---",
        "",
    ]
    return "\n".join(parts)


def render_report(results: list[tuple[dict, dict, dict]]) -> str:
    n = len(results)
    now = datetime.now(timezone.utc)

    b_latencies = [b.get("elapsed_ms", 0) for _, b, _ in results if "_error" not in b]
    g_latencies = [g.get("elapsed_ms", 0) for _, _, g in results if "_error" not in g]
    avg_b = int(sum(b_latencies) / len(b_latencies)) if b_latencies else 0
    avg_g = int(sum(g_latencies) / len(g_latencies)) if g_latencies else 0

    header = f"""\
# Smoke Eval — {now.strftime("%Y-%m-%d")}

System: 5-node multi-agent graph (Planner → Searcher → FactChecker → Writer → Critic) vs single-node RAG baseline.
Questions: {n}
Endpoints: http://localhost:8000/research (baseline), http://localhost:8000/research/graph (graph)

## Rubric (per answer, 0-2 each, 8 max)
- **Groundedness:** Are all claims in the answer supported by the cited sources? (2 = fully grounded; 1 = mostly; 0 = significant unsupported claims)
- **Citation accuracy:** Do inline [N] references actually point to the source they claim to? (2 = all correct; 1 = minor errors; 0 = significant misattribution)
- **Completeness:** Does the answer address all parts of what was asked? (2 = fully; 1 = mostly; 0 = significant aspects missing)
- **Hallucination check:** Are there any fabricated claims (made-up facts, invented sources, false specifics)? (2 = none; 1 = minor; 0 = serious fabrication)

## How to use
1. Read both answers for each question carefully.
2. Fill in the score table inline (0, 1, or 2 per criterion).
3. Write 1-2 sentences in the notes column when something stands out.
4. After grading all questions, fill in the summary at the bottom of this report.

---

"""

    body = "\n".join(_render_question(q, b, g) for q, b, g in results)

    footer = f"""\
## Summary (fill in after grading)

- **Baseline total score:**  / {n * 8}
- **Graph total score:**  / {n * 8}
- **Graph wins:**  questions
- **Baseline wins:**  questions
- **Ties:**  questions
- **Average latency — baseline:** {avg_b}ms
- **Average latency — graph:** {avg_g}ms

### Where graph helped most
_1-2 sentences_

### Where graph didn't help (or hurt)
_1-2 sentences_

### Overall takeaway
_1 paragraph: which question types benefit from multi-agent, which don't, what to focus on for Week 4_
"""

    return header + body + "\n" + footer


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main(questions_path: Path, output_path: Path) -> None:
    questions = load_questions(questions_path)

    active: list[dict] = []
    for q in questions:
        if q["text"].startswith("PLACEHOLDER"):
            print(f"[SKIP] {q['id']}: placeholder — edit {questions_path.name} before running")
        else:
            active.append(q)

    n_skipped = len(questions) - len(active)
    print(f"Eval: {len(active)} questions to run, {n_skipped} skipped")

    n_errors = 0
    results: list[tuple[dict, dict, dict]] = []

    async with httpx.AsyncClient(timeout=TIMEOUT_S) as client:
        for i, q in enumerate(active, 1):
            print(f"  [{i}/{len(active)}] {q['id']}: {q['text'][:70]}...")

            baseline = await call_endpoint(client, "/research", q["text"])
            if "_error" in baseline:
                print(f"    baseline ERROR: {baseline['_error']}")
                n_errors += 1

            graph = await call_endpoint(client, "/research/graph", q["text"])
            if "_error" in graph:
                print(f"    graph   ERROR: {graph['_error']}")
                n_errors += 1

            results.append((q, baseline, graph))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_report(results), encoding="utf-8")
    print(f"Eval complete. {len(active)} questions answered, {n_errors} errors. Report: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smoke eval for the research assistant API.")
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS, metavar="PATH")
    parser.add_argument("--output", type=Path, default=None, metavar="PATH")
    args = parser.parse_args()

    if args.output is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.output = DEFAULT_REPORTS / f"smoke-eval-{ts}.md"

    asyncio.run(main(args.questions, args.output))
