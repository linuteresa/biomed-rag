"""Pinecone hybrid vector store: namespaces, sparse-dense upsert, metadata filter.

Why a hand-rolled store over the SDK rather than LlamaIndex's PineconeVectorStore:
it keeps the hybrid mechanics explicit and fully under our control — the alpha
weighting between dense and BM25, the per-doc_type namespaces, and the metadata
filter language — which is exactly the part this project is meant to showcase.
LlamaIndex still owns ingestion/orchestration and node construction upstream.

Hybrid search requires a `dotproduct` index; dense and sparse contributions are
combined at query time via `embeddings.hybrid_scale`.
"""
from __future__ import annotations

from typing import Iterable, Sequence

from llama_index.core.schema import BaseNode

from .config import CONFIG, Config
from .embeddings import hybrid_scale

# Pinecone caps metadata at 40 KB/vector; keep stored node text bounded.
_MAX_TEXT_CHARS = 8000


def _node_metadata(node: BaseNode) -> dict:
    """Pinecone-safe metadata for a node: its own metadata + the node text."""
    md = {k: v for k, v in node.metadata.items() if v not in (None, "", [], {})}
    text = (node.get_content() or "")[:_MAX_TEXT_CHARS]
    md["text"] = text
    return md


class PineconeHybridIndex:
    def __init__(self, config: Config = CONFIG):
        from pinecone import Pinecone

        self.cfg = config
        self.pc = Pinecone(api_key=config.pinecone.api_key)
        self.index_name = config.pinecone.index_name

    # ----------------------------------------------------------------- setup
    def ensure_index(self, dimension: int | None = None) -> None:
        """Create the serverless index if it doesn't exist (idempotent)."""
        from pinecone import ServerlessSpec

        dim = dimension or self.cfg.embed.dim
        existing = {i["name"] for i in self.pc.list_indexes()}
        if self.index_name in existing:
            return
        self.pc.create_index(
            name=self.index_name,
            dimension=dim,
            metric=self.cfg.pinecone.metric,  # dotproduct, required for hybrid
            spec=ServerlessSpec(
                cloud=self.cfg.pinecone.cloud, region=self.cfg.pinecone.region
            ),
        )

    def _index(self):
        return self.pc.Index(self.index_name)

    # ---------------------------------------------------------------- upsert
    def upsert_nodes(
        self,
        nodes: Sequence[BaseNode],
        dense_embed_model,
        sparse_encoder,
        namespace: str,
        batch_size: int = 100,
    ) -> int:
        """Embed nodes (dense + sparse) and upsert them into one namespace."""
        index = self._index()
        total = 0
        for start in range(0, len(nodes), batch_size):
            batch = nodes[start : start + batch_size]
            texts = [n.get_content() for n in batch]
            dense = dense_embed_model.get_text_embedding_batch(texts, show_progress=False)
            sparse = sparse_encoder.encode_documents(texts)
            vectors = [
                {
                    "id": n.node_id,
                    "values": d,
                    "sparse_values": {
                        "indices": s["indices"],
                        "values": s["values"],
                    },
                    "metadata": _node_metadata(n),
                }
                for n, d, s in zip(batch, dense, sparse)
            ]
            index.upsert(vectors=vectors, namespace=namespace)
            total += len(vectors)
        return total

    # ----------------------------------------------------------------- query
    def query(
        self,
        query_text: str,
        dense_embed_model,
        sparse_encoder,
        top_k: int = 10,
        alpha: float = 0.5,
        namespace: str | None = None,
        metadata_filter: dict | None = None,
    ) -> list[dict]:
        """Hybrid query: dense + BM25 sparse, blended by alpha, filtered by metadata.

        `metadata_filter` uses Pinecone's filter language, e.g.
            {"year": {"$gte": 2020}, "doc_type": {"$eq": "abstract"}}
        """
        dense = dense_embed_model.get_query_embedding(query_text)
        sparse = sparse_encoder.encode_query(query_text)
        scaled_dense, scaled_sparse = hybrid_scale(dense, sparse, alpha)

        res = self._index().query(
            vector=scaled_dense,
            sparse_vector=scaled_sparse,
            top_k=top_k,
            namespace=namespace,
            filter=metadata_filter,
            include_metadata=True,
        )
        out: list[dict] = []
        for m in res.get("matches", []):
            md = m.get("metadata", {}) or {}
            out.append(
                {
                    "id": m.get("id"),
                    "score": m.get("score"),
                    "text": md.get("text", ""),
                    "metadata": {k: v for k, v in md.items() if k != "text"},
                }
            )
        return out

    def stats(self) -> dict:
        return self._index().describe_index_stats().to_dict()


def namespace_for(doc_type: str) -> str:
    """Map a document type to its Pinecone namespace."""
    return doc_type or CONFIG.DOC_TYPE_ABSTRACT
