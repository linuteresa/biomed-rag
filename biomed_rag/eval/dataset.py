"""Benchmark loading + validation for the retrieval eval harness.

The labelled set lives in `data/benchmarks/`:

* `corpus.xml`   — a synthetic PubMed-XML fixture (16 records, distinct topics).
* `qrels.jsonl`  — one query per line:
      {"query_id": "q03",
       "query": "SGLT2 inhibitors and hospitalization for heart failure",
       "relevant": {"91000002": 2}}     # pmid -> graded relevance (>=1 relevant)

Labels are keyed by **PMID**, not node id, so they stay valid when the chunking
strategy in `parse.py` changes — the harness maps each retrieved node back to
its source PMID before scoring.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Sequence

from ..ingest import BiomedDocument, load_records_from_file, records_to_documents

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH_DIR = os.path.normpath(os.path.join(_HERE, "..", "..", "data", "benchmarks"))
DEFAULT_CORPUS = os.path.join(_BENCH_DIR, "corpus.xml")
DEFAULT_QRELS = os.path.join(_BENCH_DIR, "qrels.jsonl")


@dataclass(frozen=True)
class Query:
    id: str
    text: str
    relevant: dict[str, float]  # pmid -> graded relevance (>= 1 means relevant)
    split: str = "easy"  # "easy" = lexical overlap with target; "hard" = paraphrase

    @property
    def relevant_pmids(self) -> set[str]:
        return {p for p, g in self.relevant.items() if g >= 1}


@dataclass
class Benchmark:
    queries: list[Query]
    name: str = "biomed-qrels"

    def __len__(self) -> int:
        return len(self.queries)

    def __iter__(self):
        return iter(self.queries)


def load_benchmark(path: str = DEFAULT_QRELS, name: str | None = None) -> Benchmark:
    queries: list[Query] = []
    seen: set[str] = set()
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                row = json.loads(raw)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({e})") from e
            qid = str(row.get("query_id") or row.get("id") or f"q{lineno}")
            if qid in seen:
                raise ValueError(f"{path}:{lineno}: duplicate query_id {qid!r}")
            seen.add(qid)
            text = (row.get("query") or row.get("text") or "").strip()
            if not text:
                raise ValueError(f"{path}:{lineno}: empty query text")
            rel_raw = row.get("relevant") or row.get("relevant_pmids") or {}
            if isinstance(rel_raw, list):  # allow a bare list of pmids (binary)
                rel = {str(p): 1.0 for p in rel_raw}
            else:
                rel = {str(p): float(g) for p, g in rel_raw.items()}
            if not rel:
                raise ValueError(f"{path}:{lineno}: query {qid!r} has no relevant docs")
            split = str(row.get("split") or "easy").strip().lower()
            queries.append(Query(id=qid, text=text, relevant=rel, split=split))
    return Benchmark(queries=queries, name=name or os.path.basename(path))


def load_corpus(
    path: str = DEFAULT_CORPUS, doc_type: str = "abstract"
) -> list[BiomedDocument]:
    """Parse the benchmark PubMed-XML corpus into BiomedDocuments."""
    records = load_records_from_file(path)
    return records_to_documents(records, doc_type=doc_type)


def validate_benchmark(
    benchmark: Benchmark, corpus: Sequence[BiomedDocument]
) -> list[str]:
    """Return a list of human-readable problems (empty == clean)."""
    corpus_pmids = {d.metadata.get("pmid") for d in corpus}
    problems: list[str] = []
    for q in benchmark.queries:
        missing = q.relevant_pmids - corpus_pmids
        if missing:
            problems.append(
                f"{q.id}: relevant pmid(s) not in corpus: {sorted(missing)}"
            )
    covered = set().union(*(q.relevant_pmids for q in benchmark.queries)) if benchmark.queries else set()
    never_relevant = corpus_pmids - covered
    if never_relevant:
        problems.append(
            f"note: {len(never_relevant)} corpus doc(s) are relevant to no query "
            f"(fine as distractors): {sorted(p for p in never_relevant if p)}"
        )
    return problems
