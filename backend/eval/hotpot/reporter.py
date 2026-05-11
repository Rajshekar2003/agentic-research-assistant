"""HotpotQA eval report renderer — formats a scores JSON into a polished markdown report.

Reads the JSON produced by scorer.py's score_run (summary aggregates + by_type breakdowns
+ per_question array) and renders it into a structured markdown document. Optionally
enriches the report with latency stats and answer text from the original runner JSONL.

Entry point:
    python -m eval.hotpot.reporter --input PATH [--run-jsonl PATH] [--output PATH]
"""

from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Refusal patterns
# ---------------------------------------------------------------------------

_REFUSAL_PATTERNS: tuple[str, ...] = (
    "couldn't verify",
    "could not verify",
    "do not contain information",
    "do not contain enough information",
    "no information available",
    "cannot be determined",
    "i don't have",
    "i do not have",
)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def is_refusal(prediction: str) -> bool:
    """Check whether a prediction string is an honest refusal.

    Uses case-insensitive substring matching against a fixed set of patterns
    that correspond to fallback messages observed in the baseline and graph
    endpoints (FactChecker's "I couldn't verify" and baseline's
    "do not contain information" family).

    Args:
        prediction: Model output string to test.

    Returns:
        True if the string contains any refusal pattern, False otherwise.
    """
    if not prediction:
        return False
    lowered = prediction.lower()
    return any(pat in lowered for pat in _REFUSAL_PATTERNS)


def select_sample_questions(scored_questions: list[dict], n: int = 4) -> list[dict]:
    """Select up to n questions that span the score distribution for qualitative review.

    Attempts to return one representative from each of four outcome categories:
    graph wins by F1, baseline wins by F1, both answered correctly, both scored 0.
    Fewer than n questions are returned when the run is small or lacks distinct
    categories — this is intentional, not a bug.

    Algorithm:
        1. Sort by (graph_f1 − baseline_f1) descending; pick the top as "graph wins."
        2. Reverse-sort remaining; pick the top as "baseline wins."
        3. From remaining, pick first where both graph_em==1 and baseline_em==1; fall
           back to highest combined F1 if no exact match exists.
        4. From remaining, pick first where both graph_f1==0.0 and baseline_f1==0.0.

    Args:
        scored_questions: List of per-question dicts from scorer.py's score_run output.
            Each dict is expected to have graph_f1, baseline_f1, graph_em, baseline_em.
        n: Maximum number of questions to return (default 4).

    Returns:
        List of up to n question dicts. Never contains duplicates.
    """
    if not scored_questions:
        return []

    def _f1(q: dict, key: str) -> float:
        v = q.get(key)
        return float(v) if v is not None else 0.0

    def _em(q: dict, key: str) -> int:
        v = q.get(key)
        return int(v) if v is not None else 0

    selected: list[dict] = []
    selected_ids: set[str] = set()

    def _add(q: dict | None) -> None:
        if q is None or q["question_id"] in selected_ids:
            return
        selected.append(q)
        selected_ids.add(q["question_id"])

    remaining = list(scored_questions)

    # 1. Graph wins (highest graph_f1 - baseline_f1)
    by_delta_desc = sorted(
        remaining,
        key=lambda q: _f1(q, "graph_f1") - _f1(q, "baseline_f1"),
        reverse=True,
    )
    _add(by_delta_desc[0] if by_delta_desc else None)
    remaining = [q for q in remaining if q["question_id"] not in selected_ids]

    # 2. Baseline wins (most negative delta)
    by_delta_asc = sorted(
        remaining,
        key=lambda q: _f1(q, "graph_f1") - _f1(q, "baseline_f1"),
    )
    _add(by_delta_asc[0] if by_delta_asc else None)
    remaining = [q for q in remaining if q["question_id"] not in selected_ids]

    # 3. Both scored well (both EM=1, or highest combined F1 as fallback)
    both_correct = [
        q for q in remaining
        if _em(q, "graph_em") == 1 and _em(q, "baseline_em") == 1
    ]
    if both_correct:
        _add(both_correct[0])
    else:
        by_combined = sorted(
            remaining,
            key=lambda q: _f1(q, "graph_f1") + _f1(q, "baseline_f1"),
            reverse=True,
        )
        _add(by_combined[0] if by_combined else None)
    remaining = [q for q in remaining if q["question_id"] not in selected_ids]

    # 4. Both failed (both F1 == 0.0)
    both_zero = [
        q for q in remaining
        if _f1(q, "graph_f1") == 0.0 and _f1(q, "baseline_f1") == 0.0
    ]
    _add(both_zero[0] if both_zero else None)

    return selected[:n]


def latency_stats(run_data: list[dict]) -> dict:
    """Compute mean, median, and p95 latency for baseline and graph endpoints.

    Args:
        run_data: List of dicts parsed from runner JSONL lines. Each dict must
            contain "baseline" and "graph" sub-dicts with "status" and "elapsed_ms"
            keys. Entries where status != "ok" are excluded from statistics.

    Returns:
        Dict with keys "baseline" and "graph", each a sub-dict containing:
            "mean_ms" (int), "median_ms" (int), "p95_ms" (int).
            All values are None when no valid entries exist for that endpoint.
    """
    def _compute(ms_values: list[int]) -> dict[str, int | None]:
        if not ms_values:
            return {"mean_ms": None, "median_ms": None, "p95_ms": None}
        s = sorted(ms_values)
        p95_idx = int(len(s) * 0.95)
        if p95_idx >= len(s):
            p95_idx = len(s) - 1
        return {
            "mean_ms": round(statistics.mean(s)),
            "median_ms": round(statistics.median(s)),
            "p95_ms": s[p95_idx],
        }

    baseline_ms = [
        r["baseline"]["elapsed_ms"]
        for r in run_data
        if r["baseline"]["status"] == "ok" and r["baseline"].get("elapsed_ms") is not None
    ]
    graph_ms = [
        r["graph"]["elapsed_ms"]
        for r in run_data
        if r["graph"]["status"] == "ok" and r["graph"].get("elapsed_ms") is not None
    ]

    return {"baseline": _compute(baseline_ms), "graph": _compute(graph_ms)}


def format_pct(x: float) -> str:
    """Format a float metric value as a 3-decimal string for use in tables.

    Args:
        x: Float value, typically an EM or F1 score in [0, 1].

    Returns:
        String formatted as "{x:.3f}".
    """
    return f"{x:.3f}"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_run_jsonl(path: Path) -> dict[str, dict]:
    """Load runner JSONL into a dict keyed by question_id."""
    result: dict[str, dict] = {}
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                entry = json.loads(stripped)
                result[entry["question_id"]] = entry
            except (json.JSONDecodeError, KeyError):
                continue
    return result


def _auto_commentary(q: dict, run_entry: dict | None) -> str:
    """Generate a one-line qualitative commentary for a sample question pair."""
    g_f1 = q.get("graph_f1") or 0.0
    b_f1 = q.get("baseline_f1") or 0.0
    g_em = q.get("graph_em") or 0
    b_em = q.get("baseline_em") or 0
    graph_pred = q.get("graph_prediction") or ""
    baseline_pred = q.get("baseline_prediction") or ""

    if is_refusal(graph_pred) and is_refusal(baseline_pred):
        return "Both refused; sources did not contain the answer."
    if g_em == 1 and b_em == 1:
        return "Both endpoints answered correctly."
    if is_refusal(graph_pred):
        return "Graph refused; baseline attempted an answer."
    if is_refusal(baseline_pred):
        return "Baseline refused; graph attempted an answer."

    delta = g_f1 - b_f1
    if delta > 0.05:
        facts_note = ""
        if run_entry:
            facts = run_entry["graph"].get("facts") or []
            facts_note = f"; FactChecker extracted {len(facts)} verified fact(s)"
        return f"Graph wins by F1{facts_note}."
    if delta < -0.05:
        return "Baseline wins by F1."
    if g_f1 == 0.0 and b_f1 == 0.0:
        return "Both scored 0; sources did not contain the answer."
    return "Scores are similar across both endpoints."


# ---------------------------------------------------------------------------
# Main renderer
# ---------------------------------------------------------------------------


def render_report(
    scores: dict,
    run_jsonl_path: Path | None = None,
    output_path: Path | None = None,
    scores_path: Path | None = None,
) -> str:
    """Render a markdown eval report from a scorer.py scores dict.

    Args:
        scores: Dict produced by scorer.py's score_run, with keys "summary",
            "by_type", and "per_question".
        run_jsonl_path: Optional path to the original runner JSONL. When provided,
            the report includes latency statistics and richer sample question details
            (answer text, elapsed time, fact counts, critic verdict, revision count).
            When omitted, those sections show placeholder text.
        output_path: If provided, the rendered markdown is written to this file
            in addition to being returned.
        scores_path: Optional path to the scores JSON file, used only in the run
            metadata section for traceability.

    Returns:
        Rendered markdown string.
    """
    summary = scores["summary"]
    by_type = scores.get("by_type", {})
    per_question = scores.get("per_question", [])

    n = summary["n_questions"]
    n_baseline_ok = summary["n_baseline_ok"]
    n_graph_ok = summary["n_graph_ok"]
    b_em = summary["baseline_em"]
    b_f1 = summary["baseline_f1"]
    g_em = summary["graph_em"]
    g_f1 = summary["graph_f1"]

    comp = by_type.get("comparison", {})
    brdg = by_type.get("bridge", {})
    n_comp = comp.get("n_questions", 0)
    n_bridge = brdg.get("n_questions", 0)

    run_data_by_id: dict[str, dict] = {}
    run_data_list: list[dict] = []
    if run_jsonl_path is not None:
        run_data_by_id = _load_run_jsonl(run_jsonl_path)
        run_data_list = list(run_data_by_id.values())

    now = datetime.now(timezone.utc)
    iso_date = now.strftime("%Y-%m-%d")
    iso_timestamp = now.strftime("%Y-%m-%dT%H:%M:%SZ")

    lines: list[str] = []

    # ---- Header ----
    lines += [
        f"# HotpotQA Eval Report — {iso_date}",
        "",
        "**System:** 5-node multi-agent graph "
        "(Planner → Searcher → FactChecker → Writer → Critic) "
        "vs single-node RAG baseline.",
        f"**Dataset:** HotpotQA dev set (distractor setting), {n} questions sampled.",
        "**Metrics:** Exact Match (EM) and token-level F1, computed by direct "
        "reproduction of HotpotQA's official hotpot_evaluate_v1.py.",
        "",
        "---",
        "",
    ]

    # ---- Headline numbers ----
    lines += [
        "## Headline numbers",
        "",
        "|              | Baseline   | Graph      |",
        "|--------------|------------|------------|",
        f"| Exact Match  | {format_pct(b_em):<10} | {format_pct(g_em):<10} |",
        f"| F1           | {format_pct(b_f1):<10} | {format_pct(g_f1):<10} |",
        "",
        f"{n} questions scored. {n_baseline_ok} baseline OK, {n_graph_ok} graph OK.",
        "",
    ]

    # ---- By question type ----
    comp_header = f"Comparison ({n_comp})"
    brdg_header = f"Bridge ({n_bridge})"
    lines += [
        "## By question type",
        "",
        f"|                  | {comp_header} | {brdg_header} |",
        f"|------------------|{'-' * (len(comp_header) + 2)}|{'-' * (len(brdg_header) + 2)}|",
        f"| Baseline EM      | {format_pct(comp.get('baseline_em', 0.0)):<{len(comp_header)}} | "
        f"{format_pct(brdg.get('baseline_em', 0.0)):<{len(brdg_header)}} |",
        f"| Baseline F1      | {format_pct(comp.get('baseline_f1', 0.0)):<{len(comp_header)}} | "
        f"{format_pct(brdg.get('baseline_f1', 0.0)):<{len(brdg_header)}} |",
        f"| Graph EM         | {format_pct(comp.get('graph_em', 0.0)):<{len(comp_header)}} | "
        f"{format_pct(brdg.get('graph_em', 0.0)):<{len(brdg_header)}} |",
        f"| Graph F1         | {format_pct(comp.get('graph_f1', 0.0)):<{len(comp_header)}} | "
        f"{format_pct(brdg.get('graph_f1', 0.0)):<{len(brdg_header)}} |",
        "",
    ]

    # ---- Latency ----
    lines.append("## Latency")
    lines.append("")
    lines += [
        "| Endpoint  | Mean (ms) | Median (ms) | P95 (ms) |",
        "|-----------|-----------|-------------|----------|",
    ]
    if run_jsonl_path is not None and run_data_list:
        stats = latency_stats(run_data_list)
        b_st = stats["baseline"]
        g_st = stats["graph"]

        def _ms(v: int | None) -> str:
            return str(v) if v is not None else "N/A"

        lines += [
            f"| Baseline  | {_ms(b_st['mean_ms']):<9} | {_ms(b_st['median_ms']):<11} | "
            f"{_ms(b_st['p95_ms']):<8} |",
            f"| Graph     | {_ms(g_st['mean_ms']):<9} | {_ms(g_st['median_ms']):<11} | "
            f"{_ms(g_st['p95_ms']):<8} |",
        ]
    else:
        lines += [
            "| Baseline  | —         | —           | —        |",
            "| Graph     | —         | —           | —        |",
            "",
            "(Latency from run JSONL. Skipped if run JSONL not provided.)",
        ]
    lines.append("")

    # ---- Honest refusals ----
    lines += [
        "## Honest refusals",
        "",
        "The HotpotQA EM/F1 metrics do not distinguish between confident-but-wrong "
        "answers and honest \"I don't know\" refusals — both score 0. Identifying "
        "refusals separately is therefore important for interpreting the headline numbers.",
        "",
        'A prediction is considered an "honest refusal" if it matches one of these '
        "patterns (case-insensitive):",
        "",
        '- "couldn\'t verify"',
        '- "could not verify"',
        '- "do not contain information"',
        '- "do not contain enough information"',
        '- "no information available"',
        '- "cannot be determined"',
        '- "I don\'t have"',
        '- "I do not have"',
        "",
    ]

    baseline_ok_qs = [
        q for q in per_question
        if q.get("baseline_status") == "ok" and q.get("baseline_prediction") is not None
    ]
    graph_ok_qs = [
        q for q in per_question
        if q.get("graph_status") == "ok" and q.get("graph_prediction") is not None
    ]
    n_baseline_refusals = sum(1 for q in baseline_ok_qs if is_refusal(q["baseline_prediction"]))
    n_graph_refusals = sum(1 for q in graph_ok_qs if is_refusal(q["graph_prediction"]))

    lines += [
        "**Refusal counts:**",
        f"- Baseline refusals: {n_baseline_refusals} / {n_baseline_ok}",
        f"- Graph refusals: {n_graph_refusals} / {n_graph_ok}",
        "",
        "**Refusal examples:**",
        "",
    ]

    graph_refusals = [
        q for q in per_question
        if q.get("graph_prediction") and is_refusal(q["graph_prediction"])
    ][:3]

    if graph_refusals:
        for i, q in enumerate(graph_refusals, 1):
            qid = q["question_id"]
            run_entry = run_data_by_id.get(qid)
            if run_entry:
                question_text = run_entry["question_text"]
            else:
                question_text = f"(run JSONL not provided; question_id={qid})"
            lines += [
                f"**Example {i}:**",
                f"- **Question:** {question_text}",
                f"- **Ground truth:** `{q['ground_truth']}`",
                f"- **Graph answer:** {q['graph_prediction']}",
                "",
            ]
    else:
        lines += ["_(No graph refusals found in this run.)_", ""]

    # ---- Sample question pairs ----
    sample = select_sample_questions(per_question, n=4)
    n_sample = len(sample)

    lines += [
        "## Sample question pairs",
        "",
        f"For qualitative inspection, here are {min(n_sample, n)} questions chosen "
        "to span the score distribution: 1 where graph clearly beat baseline by F1, "
        "1 where baseline beat graph, 1 where both scored well, 1 where both scored 0 "
        "(refusal or genuine failure).",
        "",
    ]

    if n_sample < 4:
        lines += [f"_Only {n_sample} distinct cases available in this run._", ""]

    if not sample:
        lines += ["_(No questions available.)_", ""]
    else:
        for i, q in enumerate(sample, 1):
            qid = q["question_id"]
            run_entry = run_data_by_id.get(qid)
            question_text = run_entry["question_text"] if run_entry else f"(question_id: {qid})"
            ground_truth = q.get("ground_truth", "")
            b_pred = q.get("baseline_prediction") or "(no answer)"
            g_pred = q.get("graph_prediction") or "(no answer)"
            b_em_val = q.get("baseline_em") or 0
            b_f1_val = q.get("baseline_f1") or 0.0
            g_em_val = q.get("graph_em") or 0
            g_f1_val = q.get("graph_f1") or 0.0

            if run_entry:
                b_elapsed = run_entry["baseline"].get("elapsed_ms", "?")
                g_elapsed = run_entry["graph"].get("elapsed_ms", "?")
                facts = run_entry["graph"].get("facts") or []
                facts_count = len(facts)
                critic_verdict = run_entry["graph"].get("critic_verdict") or "N/A"
                revisions = run_entry["graph"].get("revisions")
                revisions_str = str(revisions) if revisions is not None else "N/A"
            else:
                b_elapsed = "?"
                g_elapsed = "?"
                facts_count = "?"
                critic_verdict = "N/A"
                revisions_str = "N/A"

            commentary = _auto_commentary(q, run_entry)

            lines += [
                f"### Q{i}: {question_text}",
                "",
                f"**Ground truth:** `{ground_truth}`",
                "",
                f"**Baseline ({b_elapsed}ms):** {b_pred}",
                "",
                f"- EM: {b_em_val} | F1: {b_f1_val:.3f}",
                "",
                f"**Graph ({g_elapsed}ms, plan_size=N/A, facts={facts_count}, "
                f"verdict={critic_verdict}, revisions={revisions_str}):** {g_pred}",
                "",
                f"- EM: {g_em_val} | F1: {g_f1_val:.3f}",
                "",
                f"_{commentary}_",
                "",
            ]

    # ---- Limitations ----
    lines += [
        "## Limitations",
        "",
        "- HotpotQA's EM and F1 do not credit honest refusals; refer to the dedicated "
        "refusal section above.",
        "- Open-web retrieval via Tavily differs from the closed-book setting common "
        "in published HotpotQA scores. Trends are interpretable but absolute numbers "
        "are not directly comparable to the original paper's baselines.",
        f"- Sample size for this run: {n} questions. Statistical significance is "
        "limited below ~50 questions.",
        "- Smoke runs may include questions with sparse or speculative source coverage; "
        "bridge questions especially may suffer when the second hop requires entity "
        "resolution from the first hop's output.",
        "",
    ]

    # ---- Run metadata ----
    run_jsonl_display = str(run_jsonl_path) if run_jsonl_path else "(not provided)"
    scores_path_display = str(scores_path) if scores_path else "(not specified)"
    lines += [
        "## Run metadata",
        "",
        f"- Source JSONL: `{run_jsonl_display}`",
        f"- Source scores: `{scores_path_display}`",
        f"- Generated: `{iso_timestamp}`",
        "",
    ]

    markdown = "\n".join(lines)

    if output_path is not None:
        output_path.write_text(markdown, encoding="utf-8")

    return markdown


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Render a HotpotQA eval scores JSON into a markdown report."
    )
    parser.add_argument(
        "--input", type=Path, required=True, metavar="PATH",
        help="Path to the scores JSON file produced by scorer.py.",
    )
    parser.add_argument(
        "--run-jsonl", type=Path, default=None, metavar="PATH",
        help="Path to the original runner JSONL (recommended for richer report).",
    )
    parser.add_argument(
        "--output", type=Path, default=None, metavar="PATH",
        help="Output markdown path. Default: input path with -scores.json replaced by -report.md.",
    )
    args = parser.parse_args()

    if args.output is None:
        stem = args.input.stem
        if stem.endswith("-scores"):
            stem = stem[: -len("-scores")]
        args.output = args.input.with_name(stem + "-report.md")

    scores_data = json.loads(args.input.read_text(encoding="utf-8"))
    per_question = scores_data.get("per_question", [])

    markdown = render_report(
        scores_data,
        run_jsonl_path=args.run_jsonl,
        output_path=args.output,
        scores_path=args.input,
    )

    graph_ok_qs = [
        q for q in per_question
        if q.get("graph_status") == "ok" and q.get("graph_prediction") is not None
    ]
    n_refusals = sum(1 for q in graph_ok_qs if is_refusal(q["graph_prediction"]))
    n = scores_data["summary"]["n_questions"]

    print(f"Report written to {args.output}. {n} questions, {n_refusals} refusals.")
