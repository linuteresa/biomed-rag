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
    """1 / rank of the first relevant document (0 if none). Mean over queries
    is MRR."""
    rel = _relevant_set(relevance)
    for i, doc in enumerate(ranked, start=1):
        if doc in rel:
            return 1.0 / i
    return 0.0


def average_precision(ranked: Sequence[str], relevance: Relevance) -> float:
    """Average of precision@k taken at every rank that holds a relevant
    document, divided by the number of relevant documents. Mean over queries
    is MAP."""
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
    """nDCG@k = DCG@k / ideal-DCG@k. Ideal ordering sorts all graded gains
    descending. Returns 0 when there is no relevant document."""
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
