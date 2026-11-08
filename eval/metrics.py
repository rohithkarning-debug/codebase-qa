"""Evaluation metrics: Recall@k, MRR, file-location accuracy, line accuracy.

Hit rule (chunk vs. gold span)
------------------------------
A retrieved chunk counts as a hit for a gold span when it is in the same file
and at least half of the *smaller* of the two spans overlaps:

    overlap(chunk, gold) / min(len(chunk), len(gold)) >= 0.5

So a 40-line window that fully contains a 10-line gold method is a hit, but a
window that only touches its last 3 lines is not. The same rule is applied to
both chunkers, so the comparison is fair.
"""
from __future__ import annotations

from statistics import mean
from typing import Iterable

MIN_OVERLAP = 0.5


def span_overlap(a_start: int, a_end: int, b_start: int, b_end: int) -> int:
    return max(0, min(a_end, b_end) - max(a_start, b_start) + 1)


def is_hit(path: str, start: int, end: int, gold) -> bool:
    """gold: iterable of objects with .path .start .end"""
    for g in gold:
        if g.path != path:
            continue
        ov = span_overlap(start, end, g.start, g.end)
        if ov and ov / min(end - start + 1, g.end - g.start + 1) >= MIN_OVERLAP:
            return True
    return False


def first_hit_rank(retrieved: list[tuple[str, int, int]], gold) -> int | None:
    """1-based rank of the first relevant chunk, or None."""
    for i, (p, s, e) in enumerate(retrieved, 1):
        if is_hit(p, s, e, gold):
            return i
    return None


def first_file_rank(retrieved: list[tuple[str, int, int]], gold_files: set[str]) -> int | None:
    for i, (p, _s, _e) in enumerate(retrieved, 1):
        if p in gold_files:
            return i
    return None


def recall_at_k(ranks: Iterable[int | None], k: int) -> float:
    ranks = list(ranks)
    return sum(1 for r in ranks if r is not None and r <= k) / max(len(ranks), 1)


def mrr(ranks: Iterable[int | None], cutoff: int = 10) -> float:
    ranks = list(ranks)
    return sum((1.0 / r) for r in ranks if r is not None and r <= cutoff) / max(len(ranks), 1)


# ------------------------------------------------------------- answer metrics
def citation_scores(citations: list[dict], gold) -> dict:
    """File-location accuracy for a generated answer.

    file_hit  - the answer cites at least one gold file
    line_hit  - a cited line range overlaps a gold span (any overlap)
    top_file  - the FIRST citation is in a gold file (what a user clicks first)
    precision - share of cited files that are gold files
    """
    gold_files = {g.path for g in gold}
    files = [c["path"] for c in citations]
    file_hit = any(f in gold_files for f in files)
    line_hit = any(
        c.get("start") and any(g.path == c["path"] and span_overlap(c["start"], c.get("end") or c["start"],
                                                                     g.start, g.end) for g in gold)
        for c in citations
    )
    uniq = list(dict.fromkeys(files))
    return {
        "file_hit": file_hit,
        "line_hit": line_hit,
        "top_file": bool(files) and files[0] in gold_files,
        "precision": (sum(f in gold_files for f in uniq) / len(uniq)) if uniq else 0.0,
        "n_citations": len(citations),
    }


def avg(values: Iterable[float]) -> float:
    vals = [float(v) for v in values]
    return round(mean(vals), 4) if vals else 0.0
