"""The checked-in benchmark must stay well-formed and consistent with the corpus."""
from biomed_rag.eval import load_benchmark, load_corpus, validate_benchmark
from biomed_rag.eval.dataset import DEFAULT_CORPUS, DEFAULT_QRELS


def test_benchmark_loads_and_is_unique():
    bench = load_benchmark(DEFAULT_QRELS)
    assert len(bench) >= 40
    ids = [q.id for q in bench.queries]
    assert len(ids) == len(set(ids)), "duplicate query_id"
    for q in bench.queries:
        assert q.text.strip()
        assert q.relevant and all(g >= 1 for g in q.relevant.values())
        assert q.split in {"easy", "hard"}


def test_corpus_parses():
    corpus = load_corpus(DEFAULT_CORPUS)
    assert len(corpus) >= 16
    pmids = [d.metadata.get("pmid") for d in corpus]
    assert len(pmids) == len(set(pmids))
    assert all(p and p.startswith("910000") for p in pmids)


def test_every_label_resolves_to_a_corpus_doc():
    bench = load_benchmark(DEFAULT_QRELS)
    corpus = load_corpus(DEFAULT_CORPUS)
    problems = [p for p in validate_benchmark(bench, corpus) if not p.startswith("note:")]
    assert problems == [], problems


def test_has_both_splits():
    bench = load_benchmark(DEFAULT_QRELS)
    splits = {q.split for q in bench.queries}
    assert splits == {"easy", "hard"}
