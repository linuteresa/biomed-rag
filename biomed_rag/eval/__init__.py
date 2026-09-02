"""Component 2 — retrieval evaluation harness.

Pure-Python, dependency-free IR metrics (`metrics`), a checked-in labelled
benchmark loader (`dataset`), a retriever-agnostic evaluation loop (`harness`),
and an offline BM25 + TF-IDF hybrid retriever (`offline`) so the whole harness
runs with no API keys, no Pinecone, and no model download.
"""
from .dataset import Benchmark, Query, load_benchmark, load_corpus, validate_benchmark
from .harness import EvalReport, compare, evaluate
from .offline import OfflineHybridRetriever

__all__ = [
    "Benchmark",
    "Query",
    "load_benchmark",
    "load_corpus",
    "validate_benchmark",
    "EvalReport",
    "compare",
    "evaluate",
    "OfflineHybridRetriever",
]
