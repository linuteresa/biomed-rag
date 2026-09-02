"""End-to-end harness tests — offline, no services, deterministic."""
from functools import partial

import pytest

from biomed_rag.eval import compare, evaluate, load_benchmark, load_corpus
from biomed_rag.eval.dataset import Benchmark, Query
from biomed_rag.eval.offline import LexicalReranker, OfflineHybridRetriever


@pytest.fixture(scope="module")
def corpus():
    return load_corpus()


@pytest.fixture(scope="module")
def bench():
    return load_benchmark()


@pytest.fixture(scope="module")
def retriever(corpus):
    return OfflineHybridRetriever.from_corpus(corpus, reranker=LexicalReranker())


def _oracle_factory(bench):
    """A retriever that returns exactly the gold docs for each query."""
    by_text = {q.text: q for q in bench.queries}

    class _R:
        def retrieve(self, query, top_k=10, **kw):
            q = by_text[query]
            from biomed_rag.retrieve import RetrievalResult

            ordered = [p for p, _g in sorted(q.relevant.items(), key=lambda kv: -kv[1])]
            return [
                RetrievalResult(id=f"{p}::n0", score=1.0, text="", metadata={"pmid": p})
                for p in ordered
            ]

    return _R()


def test_oracle_scores_are_perfect(bench):
    rep = evaluate(_oracle_factory(bench).retrieve, bench, k_values=(1, 5, 10))
    assert rep.aggregate["recall@10"] == pytest.approx(1.0)
    assert rep.aggregate["mrr"] == pytest.approx(1.0)
    assert rep.aggregate["map"] == pytest.approx(1.0)
    assert rep.aggregate["ndcg@10"] == pytest.approx(1.0)


def test_adversarial_retriever_scores_zero(bench, corpus):
    from biomed_rag.retrieve import RetrievalResult

    all_pmids = {d.metadata["pmid"] for d in corpus}

    class _Bad:
        def retrieve(self, query, top_k=10, **kw):
            q = next(x for x in bench.queries if x.text == query)
            wrong = sorted(all_pmids - q.relevant_pmids)[:top_k]
            return [
                RetrievalResult(id=p, score=0.1, text="", metadata={"pmid": p})
                for p in wrong
            ]

    rep = evaluate(_Bad().retrieve, bench, k_values=(1, 5, 10))
    assert rep.aggregate["recall@10"] == 0.0
    assert rep.aggregate["mrr"] == 0.0


def test_offline_retriever_easy_split_is_strong(retriever, bench):
    rep = evaluate(partial(retriever.retrieve, alpha=0.5), bench, k_values=(1, 5, 10))
    assert set(rep.by_split) == {"easy", "hard"}
    assert rep.by_split["easy"]["recall@10"] >= 0.9
    # the paraphrase split is meant to be hard for a lexical retriever
    assert rep.by_split["hard"]["recall@10"] < rep.by_split["easy"]["recall@10"]


def test_alpha_endpoints_and_blend_all_run(retriever, bench):
    for a in (0.0, 0.5, 1.0):
        rep = evaluate(partial(retriever.retrieve, alpha=a), bench, k_values=(5,))
        assert 0.0 <= rep.aggregate["recall@5"] <= 1.0
    with pytest.raises(ValueError):
        retriever.retrieve("x", alpha=1.5)


def test_metadata_filter_year(retriever):
    lo = retriever.retrieve("macrophage polarization", top_k=20)
    hi = retriever.retrieve(
        "macrophage polarization", top_k=20, metadata_filter={"year": {"$gte": 2021}}
    )
    assert hi, "filter removed everything"
    assert all(m.metadata.get("year", 0) >= 2021 for m in hi)
    assert len(hi) <= len(lo)


def test_rerank_plumbing_changes_nothing_illegal(retriever, bench):
    base = evaluate(partial(retriever.retrieve, alpha=0.5), bench, k_values=(5, 10))
    rr = evaluate(
        partial(retriever.retrieve, alpha=0.5, rerank=True), bench, k_values=(5, 10)
    )
    # same recall pool (reranker only reorders), metrics stay in range
    assert rr.aggregate["recall@10"] == pytest.approx(base.aggregate["recall@10"], abs=1e-9)
    assert 0.0 <= rr.aggregate["ndcg@10"] <= 1.0


def test_compare_renders():
    b = Benchmark([Query("q1", "il-6 stat3 macrophage", {"91000001": 2})])
    r = OfflineHybridRetriever.from_corpus(load_corpus())
    reps = {f"a={a}": evaluate(partial(r.retrieve, alpha=a), b, k_values=(3,)) for a in (0.0, 1.0)}
    txt = compare(reps, metrics=("recall@3", "mrr"), baseline="a=0.0")
    assert "a=1.0" in txt and "Δrecall@3" in txt
