"""Dense + sparse encoders for hybrid retrieval.

DenseEmbedder is the seam for the retrieval fine-tuning component: today it
loads a PubMed-domain bi-encoder from the HF hub; swapping `EMBED_MODEL` (or
passing a local path) to your contrastively fine-tuned checkpoint is the only
change needed to serve fine-tuned embeddings from the same index code.

SparseEncoder is a BM25 encoder (pinecone-text). BM25 needs corpus statistics
(document frequencies) to weight terms, so it must be `.fit()` on the corpus
before encoding, and the fitted params are saved alongside the index so queries
use the same IDF weights.
"""
from __future__ import annotations

import json
from typing import Sequence

from .config import CONFIG


# --------------------------------------------------------------------------- #
# Dense
# --------------------------------------------------------------------------- #
def get_dense_embed_model(model_name: str | None = None):
    """Return a LlamaIndex embedding model (HuggingFace bi-encoder).

    Imported lazily so that importing this module (e.g. in --dry-run or tests)
    does not require torch / sentence-transformers to be installed.
    """
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    name = model_name or CONFIG.embed.model
    return HuggingFaceEmbedding(model_name=name, normalize=True)


# --------------------------------------------------------------------------- #
# Sparse (BM25)
# --------------------------------------------------------------------------- #
class SparseEncoder:
    """Thin wrapper over pinecone_text BM25Encoder with fit/encode/save/load."""

    def __init__(self):
        from pinecone_text.sparse import BM25Encoder

        self._enc = BM25Encoder()
        self._fitted = False

    def fit(self, corpus: Sequence[str]) -> "SparseEncoder":
        self._enc.fit(list(corpus))
        self._fitted = True
        return self

    def encode_documents(self, texts: Sequence[str]) -> list[dict]:
        self._require_fit()
        return self._enc.encode_documents(list(texts))

    def encode_query(self, text: str) -> dict:
        self._require_fit()
        return self._enc.encode_queries([text])[0]

    def save(self, path: str) -> None:
        self._require_fit()
        self._enc.dump(path)

    def load(self, path: str) -> "SparseEncoder":
        self._enc.load(path)
        self._fitted = True
        return self

    def _require_fit(self) -> None:
        if not self._fitted:
            raise RuntimeError(
                "SparseEncoder must be .fit(corpus) or .load(path) before encoding."
            )


# --------------------------------------------------------------------------- #
# Hybrid scaling (convex combination of dense & sparse by alpha)
# --------------------------------------------------------------------------- #
def hybrid_scale(dense: list[float], sparse: dict, alpha: float) -> tuple[list[float], dict]:
    """Scale dense and sparse vectors by alpha for a hybrid dotproduct query.

    alpha=1.0 -> pure dense, alpha=0.0 -> pure sparse (BM25). This is the
    weighting scheme from Pinecone's hybrid-search guide; it works because a
    dotproduct index scores dense and sparse contributions additively.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    scaled_dense = [v * alpha for v in dense]
    scaled_sparse = {
        "indices": sparse["indices"],
        "values": [v * (1.0 - alpha) for v in sparse["values"]],
    }
    return scaled_dense, scaled_sparse


def save_encoder_meta(path: str, model_name: str, dim: int) -> None:
    with open(path, "w") as fh:
        json.dump({"embed_model": model_name, "embed_dim": dim}, fh, indent=2)
