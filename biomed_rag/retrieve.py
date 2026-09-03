"""Hybrid retrieval + optional cross-encoder re-ranking.

Pipeline: hybrid (dense+BM25) search returns a candidate pool from Pinecone;
an optional cross-encoder re-ranks that pool by jointly scoring (query, passage)
pairs — the "retrieve wide, re-rank precise" pattern. The re-ranker is the hook
where the recall@k -> precision lift is measured in the evaluation component.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

from .config import CONFIG, Config
from .embeddings import SparseEncoder, get_dense_embed_model
from .index import PineconeHybridIndex, namespace_for
from .trace import get_logger

log = get_logger(__name__)


@dataclass
class RetrievalResult:
    id: str
    score: float
    text: str
    metadata: dict


class CrossEncoderReranker:
    """Cross-encoder re-ranker (lazy-loaded). Default is a general MS-MARCO
    model; swap for a biomedical cross-encoder (e.g. ncbi/MedCPT-Cross-Encoder)
    via the constructor for in-domain gains."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        log.info("loading cross-encoder reranker: %s (may download weights)", model_name)
        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, results: list[RetrievalResult], top_n: int
    ) -> list[RetrievalResult]:
        if not results:
            return results
        log.debug("rerank: scoring %d (query, passage) pair(s), keep top %d",
                  len(results), top_n)
        pairs = [(query, r.text) for r in results]
        scores = self.model.predict(pairs)
        reranked = sorted(
            zip(results, scores), key=lambda rs: float(rs[1]), reverse=True
        )
        out = []
        for r, s in reranked[:top_n]:
            out.append(RetrievalResult(r.id, float(s), r.text, r.metadata))
        return out


def build_metadata_filter(
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    journal: Optional[str] = None,
    mesh: Optional[str] = None,
    doc_type: Optional[str] = None,
) -> Optional[dict]:
    """Compose a Pinecone metadata filter from common biomedical facets."""
    f: dict = {}
    year: dict = {}
    if year_min is not None:
        year["$gte"] = year_min
    if year_max is not None:
        year["$lte"] = year_max
    if year:
        f["year"] = year
    if journal:
        f["journal"] = {"$eq": journal}
    if mesh:
        f["mesh_terms"] = {"$in": [mesh]}
    if doc_type:
        f["doc_type"] = {"$eq": doc_type}
    log.debug("build_metadata_filter -> %s", f or None)
    return f or None


class HybridRetriever:
    def __init__(
        self,
        index: PineconeHybridIndex,
        dense_embed_model,
        sparse_encoder: SparseEncoder,
        reranker: Optional[CrossEncoderReranker] = None,
    ):
        self.index = index
        self.dense = dense_embed_model
        self.sparse = sparse_encoder
        self.reranker = reranker

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        namespace: Optional[str] = None,
        metadata_filter: Optional[dict] = None,
        rerank: bool = False,
        candidate_pool: int = 40,
    ) -> list[RetrievalResult]:
        # When re-ranking, retrieve a wider candidate pool then trim to top_k.
        k = max(candidate_pool, top_k) if (rerank and self.reranker) else top_k
        log.debug("HybridRetriever.retrieve: %r top_k=%d alpha=%.2f rerank=%s pool=%d",
                  query, top_k, alpha, bool(rerank and self.reranker), k)
        raw = self.index.query(
            query_text=query,
            dense_embed_model=self.dense,
            sparse_encoder=self.sparse,
            top_k=k,
            alpha=alpha,
            namespace=namespace,
            metadata_filter=metadata_filter,
        )
        results = [
            RetrievalResult(r["id"], r["score"], r["text"], r["metadata"]) for r in raw
        ]
        if rerank and self.reranker:
            results = self.reranker.rerank(query, results, top_n=top_k)
        log.debug("HybridRetriever.retrieve -> %d result(s)", len(results))
        return results


def build_retriever(
    artifacts_dir: str = "data/cache",
    with_reranker: bool = False,
    config: Config = CONFIG,
) -> HybridRetriever:
    """Wire up a retriever from config: dense model, fitted BM25, Pinecone.

    Expects `bm25.json` (written by scripts/ingest.py) in `artifacts_dir`.
    """
    log.debug("build_retriever: artifacts_dir=%s with_reranker=%s", artifacts_dir, with_reranker)
    dense = get_dense_embed_model(config.embed.model)
    sparse = SparseEncoder()
    bm25_path = os.path.join(artifacts_dir, "bm25.json")
    if not os.path.exists(bm25_path):
        raise FileNotFoundError(
            f"Fitted BM25 params not found at {bm25_path}. Run scripts/ingest.py first."
        )
    log.debug("build_retriever: loading fitted BM25 from %s", bm25_path)
    sparse.load(bm25_path)
    index = PineconeHybridIndex(config)
    reranker = CrossEncoderReranker() if with_reranker else None
    log.debug("build_retriever: ready (reranker=%s)", bool(reranker))
    return HybridRetriever(index, dense, sparse, reranker)
