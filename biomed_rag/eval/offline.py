"""Offline hybrid retriever — BM25 + TF-IDF cosine, blended by alpha.

Why this exists: the production `HybridRetriever` needs Pinecone, a downloaded
bi-encoder, and API keys. To make the eval harness runnable anywhere (CI, a
laptop with no network), this module re-implements the *same* retrieve()
contract with two pure-Python lexical channels:

* sparse channel  = Okapi BM25            (mirrors `embeddings.SparseEncoder`)
* "dense" channel = TF-IDF cosine         (a deterministic stand-in for the
                                            bi-encoder's semantic similarity)

`alpha` blends them exactly like `embeddings.hybrid_scale`: alpha=1.0 -> pure
TF-IDF, alpha=0.0 -> pure BM25. Each channel's scores are min-max normalised
over the candidate set before blending so neither dominates by scale. Swap in
`retrieve.build_retriever()` for the real dense+Pinecone path with no change to
the harness.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional, Sequence

from ..config import CONFIG
from ..ingest import BiomedDocument
from ..retrieve import RetrievalResult
from ..trace import get_logger

log = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9\-]*")
_STOP = frozenset(
    "a an and are as at be by for from has have in into is it its of on or that "
    "the their this to was were which with we our study results was".split()
)


def tokenize(text: str) -> list[str]:
    return [
        t
        for t in _TOKEN_RE.findall((text or "").lower())
        if len(t) > 1 and t not in _STOP
    ]


def _minmax(scores: list[float]) -> list[float]:
    if not scores:
        return scores
    lo, hi = min(scores), max(scores)
    if hi - lo < 1e-12:
        return [0.0 for _ in scores]
    return [(s - lo) / (hi - lo) for s in scores]


class _BM25:
    """Okapi BM25 over a fixed corpus of pre-tokenised documents."""

    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [list(d) for d in docs]
        self.N = len(self.docs)
        self.dl = [len(d) for d in self.docs]
        self.avgdl = (sum(self.dl) / self.N) if self.N else 0.0
        self.tf: list[Counter] = [Counter(d) for d in self.docs]
        df: Counter = Counter()
        for tfi in self.tf:
            df.update(tfi.keys())
        # BM25+ style idf floor keeps very common terms weakly positive.
        self.idf = {
            t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()
        }

    def scores(self, query_tokens: Sequence[str]) -> list[float]:
        q = [t for t in query_tokens if t in self.idf]
        out = [0.0] * self.N
        for i in range(self.N):
            tfi, dl = self.tf[i], self.dl[i]
            if not tfi:
                continue
            s = 0.0
            denom_norm = self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            for t in q:
                f = tfi.get(t, 0)
                if f:
                    s += self.idf[t] * (f * (self.k1 + 1)) / (f + denom_norm)
            out[i] = s
        return out


class _TfidfCosine:
    """L2-normalised TF-IDF vectors with cosine similarity."""

    def __init__(self, docs: Sequence[Sequence[str]]):
        self.N = len(docs)
        df: Counter = Counter()
        for d in docs:
            df.update(set(d))
        self.idf = {t: math.log((self.N + 1) / (n + 1)) + 1.0 for t, n in df.items()}
        self.vecs = [self._vec(d) for d in docs]

    def _vec(self, tokens: Sequence[str]) -> dict[str, float]:
        tf = Counter(tokens)
        v = {t: (1 + math.log(f)) * self.idf.get(t, 0.0) for t, f in tf.items()}
        norm = math.sqrt(sum(w * w for w in v.values())) or 1.0
        return {t: w / norm for t, w in v.items()}

    def scores(self, query_tokens: Sequence[str]) -> list[float]:
        qv = self._vec(query_tokens)
        return [sum(w * dv.get(t, 0.0) for t, w in qv.items()) for dv in self.vecs]


class LexicalReranker:
    """Deterministic, model-free reranker: orders candidates by the share of
    query tokens contained in the passage (query-coverage), tie-broken by the
    original hybrid score. Stands in for `retrieve.CrossEncoderReranker` when
    no model is available; same `.rerank()` signature."""

    name = "lexical"

    def rerank(
        self, query: str, results: list[RetrievalResult], top_n: int
    ) -> list[RetrievalResult]:
        if not results:
            return results
        q = set(tokenize(query))
        scored = []
        for r in results:
            passage = set(tokenize(r.text))
            coverage = (len(q & passage) / len(q)) if q else 0.0
            scored.append((coverage, r.score, r))
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [
            RetrievalResult(r.id, float(cov), r.text, r.metadata)
            for cov, _s, r in scored[:top_n]
        ]


class OfflineHybridRetriever:
    """Same retrieve() contract as `retrieve.HybridRetriever`, no services."""

    def __init__(
        self,
        node_ids: Sequence[str],
        texts: Sequence[str],
        metadatas: Sequence[dict],
        reranker: Optional[object] = None,
    ):
        self.ids = list(node_ids)
        self.texts = list(texts)
        self.metadatas = [dict(m) for m in metadatas]
        toks = [tokenize(t) for t in self.texts]
        log.info("OfflineHybridRetriever: indexing %d node(s) (BM25 + TF-IDF)", len(self.ids))
        self._bm25 = _BM25(toks)
        self._tfidf = _TfidfCosine(toks)
        self.reranker = reranker

    # ----------------------------------------------------------------- build
    @classmethod
    def from_corpus(
        cls,
        corpus: Sequence[BiomedDocument],
        reranker: Optional[object] = None,
        **build_nodes_kwargs,
    ) -> "OfflineHybridRetriever":
        from ..parse import build_nodes

        log.debug("OfflineHybridRetriever.from_corpus: %d document(s)", len(corpus))
        nodes = build_nodes(corpus, **build_nodes_kwargs)
        ids, texts, metas = [], [], []
        for n in nodes:
            ids.append(n.node_id)
            texts.append(n.get_content() or "")
            metas.append(dict(n.metadata))
        return cls(ids, texts, metas, reranker=reranker)

    # ---------------------------------------------------------- filter + query
    @staticmethod
    def _passes_filter(md: dict, mf: Optional[dict]) -> bool:
        if not mf:
            return True
        for key, cond in mf.items():
            val = md.get(key)
            if isinstance(cond, dict):
                for op, target in cond.items():
                    if op == "$eq" and val != target:
                        return False
                    if op == "$ne" and val == target:
                        return False
                    if op == "$gte" and not (val is not None and val >= target):
                        return False
                    if op == "$lte" and not (val is not None and val <= target):
                        return False
                    if op == "$gt" and not (val is not None and val > target):
                        return False
                    if op == "$lt" and not (val is not None and val < target):
                        return False
                    if op == "$in":
                        pool = val if isinstance(val, list) else [val]
                        if not set(pool) & set(target):
                            return False
            elif val != cond:
                return False
        return True

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        namespace: Optional[str] = None,  # accepted for API parity; unused offline
        metadata_filter: Optional[dict] = None,
        rerank: bool = False,
        candidate_pool: int = 40,
    ) -> list[RetrievalResult]:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        qtok = tokenize(query)
        dense = _minmax(self._tfidf.scores(qtok))
        sparse = _minmax(self._bm25.scores(qtok))
        blended = [alpha * d + (1 - alpha) * s for d, s in zip(dense, sparse)]
        log.debug(
            "offline.retrieve: %r alpha=%.2f top_k=%d rerank=%s | query tokens=%s",
            query, alpha, top_k, bool(rerank and self.reranker), qtok,
        )

        order = sorted(range(len(self.ids)), key=lambda i: blended[i], reverse=True)
        k = max(candidate_pool, top_k) if (rerank and self.reranker) else top_k
        results: list[RetrievalResult] = []
        for i in order:
            if blended[i] <= 0:
                break
            md = self.metadatas[i]
            if not self._passes_filter(md, metadata_filter):
                continue
            results.append(
                RetrievalResult(
                    id=self.ids[i],
                    score=float(blended[i]),
                    text=self.texts[i],
                    metadata={k2: v for k2, v in md.items() if k2 != "text"},
                )
            )
            if len(results) >= k:
                break

        if rerank and self.reranker:
            before = [r.id for r in results[:top_k]]
            results = self.reranker.rerank(query, results, top_n=top_k)
            log.debug("offline.retrieve: reranked top-%d %s -> %s",
                      top_k, before, [r.id for r in results])
        log.debug("offline.retrieve -> %d result(s); top score=%.4f",
                  len(results), results[0].score if results else 0.0)
        return results


def namespace_for(doc_type: str) -> str:
    return doc_type or CONFIG.DOC_TYPE_ABSTRACT
