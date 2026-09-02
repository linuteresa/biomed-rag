"""Grounded-generation pipeline + grounding metrics — offline, ExtractiveLLM."""
import pytest

from biomed_rag.eval import load_benchmark, load_corpus
from biomed_rag.eval.offline import OfflineHybridRetriever
from biomed_rag.generate import AnswerPipeline, ExtractiveLLM
from biomed_rag.generate.evaluate import (
    citation_support,
    context_recall,
    evaluate_generation,
    faithfulness,
    has_hallucinated_citation,
)
from biomed_rag.generate.prompt import build_messages, format_contexts


@pytest.fixture(scope="module")
def pipe():
    retr = OfflineHybridRetriever.from_corpus(load_corpus())
    return AnswerPipeline.build(retriever=retr, llm=ExtractiveLLM(), top_k=5, alpha=0.5)


def test_answer_is_grounded_and_cited(pipe):
    ans = pipe.answer("does IL-6 drive M2 macrophage polarization through STAT3?")
    assert ans.text.strip()
    assert "91000001" in ans.context_pmids
    # every cited PMID was actually retrieved
    assert set(ans.citations) <= set(ans.context_pmids)
    assert not ans.hallucinated_citations
    assert faithfulness(ans.text, ans.contexts) >= 0.8
    assert citation_support(ans.text, ans.context_pmids) == pytest.approx(1.0)


def test_abstains_when_context_is_irrelevant(pipe):
    ans = pipe.answer("describe the castling rules and the en passant move in chess")
    assert "do not contain enough information" in ans.text.lower()
    # abstention is treated as faithful / supported, not a hallucination
    assert faithfulness(ans.text, ans.contexts) == 1.0
    assert not ans.hallucinated_citations


def test_fabricated_answer_scores_badly():
    from biomed_rag.retrieve import RetrievalResult

    ctx = [RetrievalResult("n1", 1.0, "Aspirin inhibits COX-1 and COX-2 enzymes.", {"pmid": "91000111"})]
    fabricated = "Aspirin cures every viral infection and reverses aging in humans [PMID:99999999]."
    assert faithfulness(fabricated, ctx) < 0.5
    assert has_hallucinated_citation(fabricated, ["91000111"]) is True
    assert citation_support(fabricated, ["91000111"]) == 0.0


def test_format_contexts_numbers_and_pmids():
    from biomed_rag.retrieve import RetrievalResult

    res = [
        RetrievalResult("a", 0.9, "text one", {"pmid": "91000001", "journal": "J", "year": 2021}),
        RetrievalResult("b", 0.8, "text two", {"pmid": "91000002"}),
    ]
    ctx, pmids = format_contexts(res)
    assert "[S1] (PMID:91000001" in ctx and "[S2] (PMID:91000002" in ctx
    assert pmids == ["91000001", "91000002"]
    msgs = build_messages("q?", res)
    assert msgs[0]["role"] == "system" and "ONLY" in msgs[0]["content"]


def test_evaluate_generation_over_benchmark(pipe):
    bench = load_benchmark()
    report = evaluate_generation(pipe, bench, top_k=5, alpha=0.5)
    agg = report["aggregate"]
    assert report["llm_backend"] == "extractive"
    assert 0.9 <= agg["faithfulness"] <= 1.0          # extractive copies context
    assert agg["hallucinated_citation"] == 0.0
    assert agg["context_recall"] >= 0.6
    assert 0.0 <= agg["context_precision"] <= 1.0
    assert 0.0 <= agg["abstention_rate"] < 0.3
    assert len(report["per_query"]) == len(bench)


def test_context_recall_math():
    assert context_recall(["1", "2", "3"], {"2", "9"}) == pytest.approx(0.5)
    assert context_recall([], {"1"}) == 0.0
