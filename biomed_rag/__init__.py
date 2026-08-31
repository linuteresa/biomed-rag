"""biomed_rag — a domain-expert RAG pipeline over biomedical literature.

Component 1 (this package): ingestion + hybrid indexing + retrieval.
    ingest      NCBI E-utilities client + PubMed XML parsing (stdlib-only core)
    parse       LlamaIndex node parsing (section-aware / hierarchical)
    embeddings  pluggable dense embedder (fine-tune-ready) + BM25 sparse encoder
    index       Pinecone hybrid store (namespaces, sparse-dense, metadata filters)
    retrieve    hybrid retriever with alpha weighting + re-ranker hook

Later components slot into the same seams: a fine-tuned bi-encoder replaces the
default in `embeddings.DenseEmbedder`, a cross-encoder fills the re-ranker hook
in `retrieve`, and a LoRA-tuned generator consumes `retrieve` output.
"""

__version__ = "0.1.0"
