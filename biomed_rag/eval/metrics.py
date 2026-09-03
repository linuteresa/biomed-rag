"""Information-retrieval metrics — pure standard library, no numpy.

Every function takes a *ranked list of document ids* (best first, already
de-duplicated to one entry per document) and a *relevance map* `{doc_id: gain}`
where `gain >= 1` marks a relevant document and larger gains mean "more
relevant" (used only by the graded nDCG). Binary metrics treat any `gain >= 1`
as relevant.

Kept deliberately framework-free and deterministic so `tests/test_metrics.py`
runs in milliseconds and the numbers are trivable by hand.
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping, Sequence

Relevance = Mapping[str, float]


def _relevant_set(relevance: Relevance) -> set[str]:
    return {d for d, g in relevance.items() if g >= 1}


def _hits(ranked: Sequence[str], relevance: Relevance, k: int) -> list[bool]:
    rel = _relevant_set(relevance)
    return [doc in rel for doc in ranked[:k]]


def precision_at_k(ranked: Sequence[str], relevance: Relevance, k: int) -> float:
    """Fraction of the top-k that are relevant. Denominator is always k."""
    if k <= 0:
        return 0.0
    return sum(_hits(ranked, relevance, k)) / k


def recall_at_k(ranked: Sequence[str], relevance: Relevance, k: int) -> float:
    """Fraction of all relevant documents that appear in the top-k."""
    rel = _relevant_set(relevance)
    if not rel:
        return 0.0
    return sum(_hits(ranked, relevance, k)) / len(rel)


def f1_at_k(ranked: Sequence[str], relevance: Relevance, k: int) -> float:
    p = precision_at_k(ranked, relevance, k)
    r = recall_at_k(ranked, relevance, k)
    return 0.0 if (p + r) == 0 else 2 * p * r / (p + r)


def hit_at_k(ranked: Sequence[str], relevance: Relevance, k: int) -> float:
    """1.0 if at least one relevant document is in the top-k, else 0.0
    (a.k.a. success@k / recall-hit)."""
    return 1.0 if any(_hits(ranked, relevance, k)) else 0.0


def reciprocal_rank(ranked: Sequence[str], relevance: Relevance) -> float:
    """First-hit score: how near the top the first relevant document lands.

    Score is 1 / (rank of the first relevant document), or 0 if none is found —
    so first place scores 1.00, second place 0.50, third 0.33. The mean of this
    over every query is the metric often abbreviated MRR (mean reciprocal rank).
    """
    rel = _relevant_set(relevance)
    for i, doc in enumerate(ranked, start=1):
        if doc in rel:
            return 1.0 / i
    return 0.0


def average_precision(ranked: Sequence[str], relevance: Relevance) -> float:
    """Average precision: running precision, sampled each time a relevant
    document is hit, then averaged over all relevant documents.

    It rewards finding every relevant document *and* ranking them high. The
    mean of this over every query is the metric often abbreviated MAP (mean
    average precision).
    """
    rel = _relevant_set(relevance)
    if not rel:
        return 0.0
    hits = 0
    running = 0.0
    for i, doc in enumerate(ranked, start=1):
        if doc in rel:
            hits += 1
            running += hits / i
    return running / len(rel)


def r_precision(ranked: Sequence[str], relevance: Relevance) -> float:
    """Precision at k = |relevant|."""
    rel = _relevant_set(relevance)
    if not rel:
        return 0.0
    return precision_at_k(ranked, relevance, len(rel))


def dcg_at_k(ranked: Sequence[str], relevance: Relevance, k: int) -> float:
    """Discounted cumulative gain with the standard log2(rank+1) discount and
    linear gains (graded relevance honoured)."""
    dcg = 0.0
    for i, doc in enumerate(ranked[:k], start=1):
        gain = relevance.get(doc, 0.0)
        if gain > 0:
            dcg += gain / math.log2(i + 1)
    return dcg


def ndcg_at_k(ranked: Sequence[str], relevance: Relevance, k: int) -> float:
    """Ranking quality in the top k, on a 0..1 scale.

    Adds up each relevant document's gain, but discounts it the further down
    the list it sits, then divides by the score of the best possible ordering.
    1.0 means "could not be ordered any better"; 0 means no relevant document
    (or none in the top k). This is the metric usually abbreviated nDCG
    (normalised discounted cumulative gain).
    """
    ideal_gains = sorted((g for g in relevance.values() if g > 0), reverse=True)
    if not ideal_gains:
        return 0.0
    idcg = sum(g / math.log2(i + 1) for i, g in enumerate(ideal_gains[:k], start=1))
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked, relevance, k) / idcg


# --------------------------------------------------------------------------- #
# Aggregation helper
# --------------------------------------------------------------------------- #
def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


# Metric families the harness iterates over. `at_k` metrics are computed for
# every k in `k_values`; `global` metrics use the full ranked list.
#
# The dict keys below are the short, standard names used everywhere the numbers
# are stored (report dicts, the JSON export, the tests). For anything shown to a
# reader, translate them through `display_name()` / `LEGEND` so the output does
# not lean on abbreviations like nDCG / MRR / MAP.
AT_K_METRICS = {
    "precision": precision_at_k,
    "recall": recall_at_k,
    "f1": f1_at_k,
    "hit": hit_at_k,
    "ndcg": ndcg_at_k,
}
GLOBAL_METRICS = {
    "mrr": reciprocal_rank,
    "map": average_precision,
    "r_precision": r_precision,
}

# Short stored key -> readable label for printed tables and reports.
DISPLAY_NAMES = {
    "precision": "precision",
    "recall": "recall",
    "f1": "f-score",
    "hit": "any-hit",
    "ndcg": "ranking-quality",
    "mrr": "first-hit-score",
    "map": "avg-precision",
    "r_precision": "precision-at-N",
}

# One-line explanations, keyed by the same short names. Printed once per run.
LEGEND = {
    "precision": "share of the returned results that are on target",
    "recall": "share of all on-target documents that were found",
    "f-score": "balance of precision and recall (harmonic mean)",
    "any-hit": "1 if at least one on-target document is in the top k, else 0",
    "ranking-quality": "0..1 — how close the ordering is to the best possible (nDCG)",
    "first-hit-score": "1 / rank of the first on-target result: #1 -> 1.00, #2 -> 0.50 (MRR)",
    "avg-precision": "rewards finding every on-target document and ranking it high (MAP)",
    "precision-at-N": "precision measured at k = the number of on-target documents",
}


def display_name(metric_key: str) -> str:
    """Translate a stored metric key to its readable label, keeping any
    ``@k`` suffix — e.g. ``"ndcg@10"`` -> ``"ranking-quality@10"``."""
    base, _, k = metric_key.partition("@")
    label = DISPLAY_NAMES.get(base, base)
    return f"{label}@{k}" if k else label


def legend_lines(metric_keys: Iterable[str]) -> list[str]:
    """Readable ``name — explanation`` lines for the given metric keys, in the
    order given, de-duplicated. Handy for printing a key above a results table.
    """
    seen: set[str] = set()
    out: list[str] = []
    for key in metric_keys:
        name = display_name(key).split("@")[0]
        if name in seen or name not in LEGEND:
            continue
        seen.add(name)
        out.append(f"{name:<16} — {LEGEND[name]}")
    return out
