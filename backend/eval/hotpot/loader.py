"""HotpotQA dataset loader — download, cache, parse, and sample the dev set.

Dataset: HotpotQA distractor setting (dev), ~7,400 multi-hop QA pairs.
License: CC BY-SA 4.0  https://hotpotqa.github.io/
Reference: Yang et al., 2018 — https://arxiv.org/abs/1809.09600
"""

import hashlib
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)

HOTPOT_DEV_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json"
HOTPOT_DEV_SHA256: str | None = None  # Fill in after first verified download.
CACHE_DIR = Path(__file__).parent / "cache"
CACHE_FILE = CACHE_DIR / "hotpot_dev_distractor_v1.json"


@dataclass(frozen=True)
class HotpotQuestion:
    """A single HotpotQA dev-set question with metadata.

    The full context passages from the dataset are deliberately excluded.
    Our system retrieves evidence via Tavily; including HotpotQA's distractor
    passages would defeat the evaluation.

    Attributes:
        id: Unique question identifier (_id field in raw JSON).
        question: The natural-language question string.
        answer: Gold answer string.
        type: Question type — "comparison" or "bridge".
        level: Difficulty level — "easy", "medium", or "hard".
        supporting_facts: Sequence of (title, sentence_idx) pairs identifying
            the gold supporting sentences.
        context_titles: Titles of the context paragraphs in the dataset (both
            supporting and distractor). Titles only — no passage text, so our
            system cannot cheat by using the dataset's own evidence.
    """

    id: str
    question: str
    answer: str
    type: Literal["comparison", "bridge"]
    level: Literal["easy", "medium", "hard"]
    # tuples (not lists) so the frozen dataclass is actually hashable — important
    # if Week 4 uses HotpotQuestion as a dict key or set element for result caching.
    supporting_facts: tuple[tuple[str, int], ...]
    context_titles: tuple[str, ...]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_sha256(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
    """urlretrieve progress callback — logs a message every ~10 MB downloaded."""
    if block_num == 0:
        return
    downloaded = block_num * block_size
    prev = (block_num - 1) * block_size
    if downloaded // (10 * 1024 * 1024) > prev // (10 * 1024 * 1024):
        mb = downloaded // (1024 * 1024)
        if total_size > 0:
            total_mb = total_size / (1024 * 1024)
            logger.info("Downloading HotpotQA: %d MB / %.0f MB ...", mb, total_mb)
        else:
            logger.info("Downloading HotpotQA: %d MB downloaded ...", mb)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def download_dataset(force: bool = False) -> Path:
    """Download the HotpotQA dev set to the local cache directory.

    Skips the download if the cached file already exists (unless force=True).
    Verifies the SHA-256 checksum if HOTPOT_DEV_SHA256 is set.

    Args:
        force: Re-download even if the file is already cached.

    Returns:
        Path to the cached JSON file.

    Raises:
        ValueError: If the downloaded file's checksum does not match
            HOTPOT_DEV_SHA256 (only when HOTPOT_DEV_SHA256 is not None).
    """
    if CACHE_FILE.exists() and not force:
        return CACHE_FILE

    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading HotpotQA dev set from %s ...", HOTPOT_DEV_URL)
    urlretrieve(HOTPOT_DEV_URL, CACHE_FILE, _reporthook)
    logger.info("Download complete: %s", CACHE_FILE)

    if HOTPOT_DEV_SHA256 is None:
        logger.warning(
            "No checksum verification — first download. "
            "Re-run with --print-hash to get the hash for integrity checks."
        )
    else:
        actual = _compute_sha256(CACHE_FILE)
        if actual != HOTPOT_DEV_SHA256:
            CACHE_FILE.unlink()
            raise ValueError(
                f"SHA-256 mismatch: expected {HOTPOT_DEV_SHA256}, got {actual}. "
                "Cached file deleted — re-run to re-download."
            )
        logger.info("Checksum verified.")

    return CACHE_FILE


def load_dataset(path: Path | None = None) -> list[HotpotQuestion]:
    """Load the HotpotQA dev set from disk, downloading it first if needed.

    Each raw entry is validated; malformed entries are skipped and logged.
    See HotpotQuestion for the parsed schema.

    Args:
        path: Path to a local JSON file. If None, calls download_dataset() to
            resolve the default cache path.

    Returns:
        List of parsed, validated HotpotQuestion instances.
    """
    if path is None:
        path = download_dataset()

    with path.open(encoding="utf-8") as f:
        raw_entries: list[dict] = json.load(f)

    questions: list[HotpotQuestion] = []
    skipped = 0

    for i, entry in enumerate(raw_entries):
        try:
            qid = entry["_id"]
            question = entry["question"]
            answer = entry["answer"]
            qtype = entry["type"]
            level = entry["level"]
            sf_raw = entry["supporting_facts"]
            context_raw = entry["context"]

            if not (isinstance(qid, str) and qid):
                raise ValueError("id is empty or not a string")
            if not (isinstance(question, str) and question):
                raise ValueError("question is empty or not a string")
            if not (isinstance(answer, str) and answer):
                raise ValueError("answer is empty or not a string")
            if qtype not in {"comparison", "bridge"}:
                raise ValueError(f"invalid type: {qtype!r}")
            if level not in {"easy", "medium", "hard"}:
                raise ValueError(f"invalid level: {level!r}")
            if not sf_raw:
                raise ValueError("supporting_facts is empty")
            for sf in sf_raw:
                if not (
                    isinstance(sf, list)
                    and len(sf) == 2
                    and isinstance(sf[0], str)
                    and isinstance(sf[1], int)
                ):
                    raise ValueError(f"malformed supporting_fact: {sf!r}")

            questions.append(
                HotpotQuestion(
                    id=qid,
                    question=question,
                    answer=answer,
                    type=qtype,
                    level=level,
                    supporting_facts=tuple(tuple(sf) for sf in sf_raw),
                    context_titles=tuple(title for title, _ in context_raw),
                )
            )
        except (KeyError, ValueError, TypeError) as exc:
            skipped += 1
            logger.debug("Skipping entry %d (%s): %s", i, entry.get("_id", "?"), exc)

    logger.info("Loaded %d questions, skipped %d malformed.", len(questions), skipped)
    return questions


def sample_questions(
    questions: list[HotpotQuestion],
    n: int,
    seed: int = 42,
    type_filter: Literal["comparison", "bridge", None] = None,
    level_filter: Literal["easy", "medium", "hard", None] = None,
) -> list[HotpotQuestion]:
    """Return a deterministic random sample from the question list.

    Args:
        questions: Full question list to sample from.
        n: Number of questions to return.
        seed: Random seed for reproducibility across runs.
        type_filter: If set, restrict sampling to questions of this type.
        level_filter: If set, restrict sampling to questions of this level.

    Returns:
        A list of n randomly sampled HotpotQuestion instances.

    Raises:
        ValueError: If n exceeds the number of questions matching the filters.
    """
    filtered = questions
    if type_filter is not None:
        filtered = [q for q in filtered if q.type == type_filter]
    if level_filter is not None:
        filtered = [q for q in filtered if q.level == level_filter]

    if n > len(filtered):
        raise ValueError(
            f"Requested {n} questions but only {len(filtered)} match the filters "
            f"(type_filter={type_filter!r}, level_filter={level_filter!r})."
        )

    return random.Random(seed).sample(filtered, n)


def get_dataset_stats(questions: list[HotpotQuestion]) -> dict:
    """Compute type/level distribution statistics for a question list.

    Useful for the runner's startup banner to confirm what was loaded.

    Args:
        questions: List of HotpotQuestion instances (typically the full dev set).

    Returns:
        Dict with keys:
            "total": total question count,
            "by_type": {"comparison": N, "bridge": N},
            "by_level": {"easy": N, "medium": N, "hard": N},
            "by_type_and_level": {(type, level): N, ...}
    """
    by_type: dict[str, int] = {"comparison": 0, "bridge": 0}
    by_level: dict[str, int] = {"easy": 0, "medium": 0, "hard": 0}
    by_type_and_level: dict[tuple[str, str], int] = {}

    for q in questions:
        by_type[q.type] = by_type.get(q.type, 0) + 1
        by_level[q.level] = by_level.get(q.level, 0) + 1
        key = (q.type, q.level)
        by_type_and_level[key] = by_type_and_level.get(key, 0) + 1

    return {
        "total": len(questions),
        "by_type": by_type,
        "by_level": by_level,
        "by_type_and_level": by_type_and_level,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="HotpotQA dev set loader — download, inspect, and sample."
    )
    parser.add_argument(
        "--print-hash",
        action="store_true",
        help=(
            "Download the dataset, compute its SHA-256, and print the hash. "
            "Use this to populate HOTPOT_DEV_SHA256 for integrity checking."
        ),
    )
    args = parser.parse_args()

    if args.print_hash:
        cached = download_dataset()
        sha256 = _compute_sha256(cached)
        print(f"SHA256: {sha256}")
        print(f'\nPaste this into loader.py:\n  HOTPOT_DEV_SHA256 = "{sha256}"')
        sys.exit(0)

    questions = load_dataset()
    stats = get_dataset_stats(questions)

    # Serialize tuple keys to strings for JSON output.
    serializable = {
        "total": stats["total"],
        "by_type": stats["by_type"],
        "by_level": stats["by_level"],
        "by_type_and_level": {
            f"{t},{lvl}": v for (t, lvl), v in stats["by_type_and_level"].items()
        },
    }
    print(json.dumps(serializable, indent=2))
