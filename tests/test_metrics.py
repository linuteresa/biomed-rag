"""Known-answer tests for the IR metrics — hand-computed, no deps."""
import math

import pytest

from biomed_rag.eval import metrics as M

# ranked doc ids, best first; relevance map pmid -> gain
RANKED = ["d1", "d2", "d3", "d4", "d5"]
BINARY = {"d2": 1, "d4": 1}          # relevant at ranks 2 and 4
GRADED = {"d1": 3, "d3": 1}          # relevant at ranks 1 and 3


def test_precision_at_k():
    assert M.precision_at_k(RANKED, BINARY, 1) == 0.0
    assert M.precision_at_k(RANKED, BINARY, 2) == 0.5
    assert M.precision_at_k(RANKED, BINARY, 4) == 0.5
    assert M.precision_at_k(RANKED, BINARY, 0) == 0.0


def test_recall_at_k():
    assert M.recall_at_k(RANKED, BINARY, 1) == 0.0
    assert M.recall_at_k(RANKED, BINARY, 2) == 0.5
    assert M.recall_at_k(RANKED, BINARY, 4) == 1.0
    assert M.recall_at_k(RANKED, {}, 5) == 0.0  # no relevant -> 0, no crash


def test_hit_and_reciprocal_rank():
    assert M.hit_at_k(RANKED, BINARY, 1) == 0.0
    assert M.hit_at_k(RANKED, BINARY, 2) == 1.0
    assert M.reciprocal_rank(RANKED, BINARY) == pytest.approx(1 / 2)
    assert M.reciprocal_rank(RANKED, {"d5": 1}) == pytest.approx(1 / 5)
    assert M.reciprocal_rank(RANKED, {"zz": 1}) == 0.0


def test_average_precision():
    # relevant at ranks 2 and 4: (1/2 + 2/4) / 2 = 0.5
    assert M.average_precision(RANKED, BINARY) == pytest.approx(0.5)
    # perfect ranking of two relevant docs -> AP 1.0
    assert M.average_precision(["a", "b", "c"], {"a": 1, "b": 1}) == pytest.approx(1.0)


def test_r_precision():
    # |relevant| = 2, precision@2 with a hit at rank 2 only -> 0.5
    assert M.r_precision(RANKED, BINARY) == pytest.approx(0.5)


def test_dcg_and_ndcg_graded():
    # DCG@3 = 3/log2(2) + 0 + 1/log2(4) = 3 + 0.5 = 3.5
    assert M.dcg_at_k(RANKED, GRADED, 3) == pytest.approx(3.5)
    # ideal order [3,1]: 3/log2(2) + 1/log2(3) = 3 + 0.6309 = 3.6309
    idcg = 3 + 1 / math.log2(3)
    assert M.ndcg_at_k(RANKED, GRADED, 3) == pytest.approx(3.5 / idcg)
    # perfect ranking -> nDCG 1.0
    assert M.ndcg_at_k(["d1", "d3"], GRADED, 3) == pytest.approx(1.0)
    assert M.ndcg_at_k(RANKED, {}, 3) == 0.0


def test_perfect_and_empty_rankings():
    rel = {"a": 2, "b": 1}
    perfect = ["a", "b", "x", "y"]
    for k in (1, 2, 5):
        assert M.recall_at_k(perfect, rel, max(k, 2)) == 1.0 or k < 2
    assert M.ndcg_at_k(perfect, rel, 5) == pytest.approx(1.0)
    assert M.average_precision(perfect, rel) == pytest.approx(1.0)
    assert M.reciprocal_rank([], rel) == 0.0
    assert M.mean([]) == 0.0
