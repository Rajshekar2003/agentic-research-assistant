"""HotpotQA answer-level scorer — exact match (EM) and F1 against gold answers.

normalize_answer, exact_match_score, and f1_score are direct reproductions of
HotpotQA's official evaluation script:
    https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py

We compute answer EM/F1 only. The official script also computes supporting-fact
EM/F1; we omit this because our system does not output supporting facts (it
outputs free text + sources).
"""

from __future__ import annotations

import argparse
import json
import re
import string
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path


# ---------------------------------------------------------------------------
# Official HotpotQA metric functions (reproduced verbatim from hotpot_evaluate_v1.py)
# ---------------------------------------------------------------------------


def normalize_answer(s: str) -> str:
    """Normalize an answer string for comparison.

    Reproduces HotpotQA's official normalization from hotpot_evaluate_v1.py.
    Applied in this exact order: lowercase, strip punctuation, strip articles,
    collapse whitespace.

    Source: https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py

    Args:
        s: Raw answer string.

    Returns:
        Normalized string suitable for EM/F1 comparison.
    """
    s = s.lower()
    s = "".join(ch if ch not in string.punctuation else " " for ch in s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


def exact_match_score(prediction: str, ground_truth: str) -> int:
    """Compute exact match between prediction and ground truth.

    Reproduces HotpotQA's official exact_match_score from hotpot_evaluate_v1.py.
    Both strings are normalized before comparison.

    Source: https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py

    Args:
        prediction: Model output string.
        ground_truth: Gold answer string.

    Returns:
        1 if normalized strings are identical, 0 otherwise.
    """
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def f1_score(prediction: str, ground_truth: str) -> tuple[float, float, float]:
    """Compute token-level F1 between prediction and ground truth.

    Reproduces HotpotQA's official f1_score from hotpot_evaluate_v1.py.
    Uses multiset intersection of normalized tokens.

    Source: https://github.com/hotpotqa/hotpot/blob/master/hotpot_evaluate_v1.py

    Args:
        prediction: Model output string.
        ground_truth: Gold answer string.

    Returns:
        Tuple of (f1, precision, recall) as floats. All three are 0.0 when there
        is no token overlap. When both strings normalize to empty, returns
        (1.0, 1.0, 1.0) — a vacuous match.
    """
    norm_pred = normalize_answer(prediction)
    norm_truth = normalize_answer(ground_truth)

    pred_tokens = norm_pred.split()
    truth_tokens = norm_truth.split()

    # Both empty → vacuous match (no tokens to compare, both said "nothing").
    if len(pred_tokens) == 0 and len(truth_tokens) == 0:
        return (1.0, 1.0, 1.0)

    # One side empty → no overlap possible.
    if len(pred_tokens) == 0 or len(truth_tokens) == 0:
        return (0.0, 0.0, 0.0)

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return (0.0, 0.0, 0.0)

    precision = num_same / len(pred_tokens)
    recall = num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return (f1, precision, recall)


# ---------------------------------------------------------------------------
# Per-question scoring dataclass
# ---------------------------------------------------------------------------


@dataclass
class QuestionScores:
    """Scores for a single HotpotQA question against both endpoints.

    Attributes:
        question_id: HotpotQA question ID.
        question_type: "comparison" or "bridge".
        question_level: Always "hard" on the dev set.
        ground_truth: Gold answer string.
        baseline_prediction: Baseline answer, or None if the endpoint errored.
        baseline_em: Exact match (0 or 1), or None if baseline errored.
        baseline_f1: Token F1, or None if baseline errored.
        graph_prediction: Graph answer, or None if the endpoint errored.
        graph_em: Exact match (0 or 1), or None if graph errored.
        graph_f1: Token F1, or None if graph errored.
        baseline_status: "ok", "http_error", "timeout", or "exception".
        graph_status: "ok", "http_error", "timeout", or "exception".
    """

    question_id: str
    question_type: str
    question_level: str
    ground_truth: str
    baseline_prediction: str | None
    baseline_em: int | None
    baseline_f1: float | None
    graph_prediction: str | None
    graph_em: int | None
    graph_f1: float | None
    baseline_status: str
    graph_status: str


# ---------------------------------------------------------------------------
# Scoring functions
# ---------------------------------------------------------------------------


def score_question(question_data: dict) -> QuestionScores:
    """Score one parsed JSONL line from the runner.

    Args:
        question_data: Dict parsed from one JSONL line produced by runner.py.
            Must contain: question_id, question_type, question_level,
            ground_truth_answer, baseline.status, baseline.answer,
            graph.status, graph.answer.

    Returns:
        QuestionScores with EM/F1 populated for ok endpoints, None for errored ones.
    """
    question_id = question_data["question_id"]
    question_type = question_data["question_type"]
    question_level = question_data["question_level"]
    ground_truth = question_data["ground_truth_answer"]

    baseline = question_data["baseline"]
    graph = question_data["graph"]

    baseline_status = baseline["status"]
    graph_status = graph["status"]

    if baseline_status == "ok" and baseline.get("answer") is not None:
        b_pred = baseline["answer"]
        b_em = exact_match_score(b_pred, ground_truth)
        b_f1, _, _ = f1_score(b_pred, ground_truth)
    else:
        b_pred = None
        b_em = None
        b_f1 = None

    if graph_status == "ok" and graph.get("answer") is not None:
        g_pred = graph["answer"]
        g_em = exact_match_score(g_pred, ground_truth)
        g_f1, _, _ = f1_score(g_pred, ground_truth)
    else:
        g_pred = None
        g_em = None
        g_f1 = None

    return QuestionScores(
        question_id=question_id,
        question_type=question_type,
        question_level=question_level,
        ground_truth=ground_truth,
        baseline_prediction=b_pred,
        baseline_em=b_em,
        baseline_f1=b_f1,
        graph_prediction=g_pred,
        graph_em=g_em,
        graph_f1=g_f1,
        baseline_status=baseline_status,
        graph_status=graph_status,
    )


def _aggregate(scores: list[QuestionScores]) -> dict:
    """Compute aggregate EM/F1 metrics for a list of QuestionScores."""
    baseline_ok = [s for s in scores if s.baseline_em is not None]
    graph_ok = [s for s in scores if s.graph_em is not None]
    return {
        "n_questions": len(scores),
        "n_baseline_ok": len(baseline_ok),
        "n_graph_ok": len(graph_ok),
        "baseline_em": (
            sum(s.baseline_em for s in baseline_ok) / len(baseline_ok)
            if baseline_ok
            else 0.0
        ),
        "baseline_f1": (
            sum(s.baseline_f1 for s in baseline_ok) / len(baseline_ok)
            if baseline_ok
            else 0.0
        ),
        "graph_em": (
            sum(s.graph_em for s in graph_ok) / len(graph_ok) if graph_ok else 0.0
        ),
        "graph_f1": (
            sum(s.graph_f1 for s in graph_ok) / len(graph_ok) if graph_ok else 0.0
        ),
    }


def score_run(jsonl_path: Path) -> dict:
    """Score all questions in a runner JSONL output file.

    Reads line-by-line for memory efficiency. Lines that fail JSON parsing are
    skipped with a warning printed to stdout.

    Args:
        jsonl_path: Path to a JSONL file produced by runner.py.

    Returns:
        Dict with keys:
            "summary": aggregate metrics across all questions,
            "by_type": same metrics broken down by "comparison" / "bridge",
            "per_question": list of QuestionScores dicts.
    """
    all_scores: list[QuestionScores] = []

    with jsonl_path.open(encoding="utf-8") as fh:
        for lineno, raw_line in enumerate(fh, start=1):
            stripped = raw_line.strip()
            if not stripped:
                continue
            try:
                data = json.loads(stripped)
            except json.JSONDecodeError as exc:
                print(f"Warning: skipping malformed JSONL line {lineno}: {exc}")
                continue
            all_scores.append(score_question(data))

    by_type: dict[str, dict] = {}
    for qtype in ("comparison", "bridge"):
        subset = [s for s in all_scores if s.question_type == qtype]
        by_type[qtype] = _aggregate(subset)

    return {
        "summary": _aggregate(all_scores),
        "by_type": by_type,
        "per_question": [asdict(s) for s in all_scores],
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Score a HotpotQA runner JSONL output file (EM + F1)."
    )
    parser.add_argument("--input", type=Path, required=True, metavar="PATH",
                        help="Path to the runner JSONL file to score.")
    parser.add_argument("--output", type=Path, default=None, metavar="PATH",
                        help="Output JSON path. Default: same dir as input, "
                             ".jsonl replaced with -scores.json.")
    args = parser.parse_args()

    if args.output is None:
        args.output = args.input.with_name(args.input.stem + "-scores.json")

    results = score_run(args.input)
    summary = results["summary"]
    by_type = results["by_type"]

    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")

    n = summary["n_questions"]
    b_em = summary["baseline_em"]
    b_f1 = summary["baseline_f1"]
    g_em = summary["graph_em"]
    g_f1 = summary["graph_f1"]

    comp = by_type["comparison"]
    brdg = by_type["bridge"]

    print(f"Scored {n} questions from {args.input}")
    print()
    print(f"{'':26}Baseline    Graph")
    print(f"  {'Exact Match':<24}{b_em:.3f}   {g_em:.3f}")
    print(f"  {'F1':<24}{b_f1:.3f}   {g_f1:.3f}")
    print()
    print("By type:")
    print(
        f"  comparison ({comp['n_questions']}):     "
        f"baseline EM={comp['baseline_em']:.3f} F1={comp['baseline_f1']:.3f}, "
        f"graph EM={comp['graph_em']:.3f} F1={comp['graph_f1']:.3f}"
    )
    print(
        f"  bridge ({brdg['n_questions']}):           "
        f"baseline EM={brdg['baseline_em']:.3f} F1={brdg['baseline_f1']:.3f}, "
        f"graph EM={brdg['graph_em']:.3f} F1={brdg['graph_f1']:.3f}"
    )
    print()
    print(f"Output: {args.output}")
