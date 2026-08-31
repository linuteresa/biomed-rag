#!/usr/bin/env python3
"""Ingest PubMed records -> parse into nodes -> (embed + upsert to Pinecone).

Examples
--------
# Offline dry run against the bundled synthetic fixture (needs only
# llama-index-core; no API keys, no Pinecone, no model download):
python scripts/ingest.py --source sample --dry-run

# Fetch live from PubMed and index into Pinecone (needs NCBI + Pinecone keys):
python scripts/ingest.py --source ncbi --query "glymphatic system amyloid" --retmax 200

# Index a previously-downloaded efetch XML file:
python scripts/ingest.py --source file --file data/cache/efetch.xml
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biomed_rag.config import CONFIG
from biomed_rag.ingest import (
    load_records_from_file,
    records_to_documents,
)

SAMPLE_FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "sample",
    "pubmed_sample.xml",
)


def get_records(args) -> list[dict]:
    if args.source == "sample":
        return load_records_from_file(SAMPLE_FIXTURE)
    if args.source == "file":
        if not args.file:
            sys.exit("--source file requires --file PATH")
        return load_records_from_file(args.file)
    if args.source == "ncbi":
        if not args.query:
            sys.exit("--source ncbi requires --query 'TERM'")
        from biomed_rag.ingest import PubMedClient

        client = PubMedClient(CONFIG)
        print(f"[ingest] esearch+efetch: {args.query!r} (retmax={args.retmax})")
        return client.search_and_fetch(args.query, retmax=args.retmax)
    sys.exit(f"unknown source: {args.source}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["sample", "ncbi", "file"], default="sample")
    p.add_argument("--query", help="PubMed query (source=ncbi)")
    p.add_argument("--file", help="path to an efetch XML file (source=file)")
    p.add_argument("--retmax", type=int, default=100)
    p.add_argument("--doc-type", default=CONFIG.DOC_TYPE_ABSTRACT,
                   choices=[CONFIG.DOC_TYPE_ABSTRACT, CONFIG.DOC_TYPE_FULLTEXT])
    p.add_argument("--artifacts-dir", default="data/cache")
    p.add_argument("--dry-run", action="store_true",
                   help="parse + build nodes + print stats; no embedding/Pinecone")
    args = p.parse_args()

    records = get_records(args)
    docs = records_to_documents(records, doc_type=args.doc_type)
    print(f"[ingest] {len(records)} records -> {len(docs)} documents "
          f"(doc_type={args.doc_type})")

    # Node building needs llama-index-core only.
    from biomed_rag.parse import build_nodes, node_summary

    nodes = build_nodes(docs)
    summary = node_summary(nodes)
    print("[ingest] node summary:")
    print(json.dumps(summary, indent=2))

    if args.dry_run:
        print("\n[ingest] --dry-run: sample node ---------------------------------")
        if nodes:
            n = nodes[0]
            print("  node_id :", n.node_id)
            print("  section :", n.metadata.get("section"))
            print("  meta    :", {k: n.metadata.get(k) for k in ("pmid", "year", "journal")})
            print("  text    :", (n.get_content() or "")[:220].replace("\n", " "), "...")
        print("\n[ingest] dry run complete — no vectors written.")
        return

    # ---- Full path: embed + upsert (needs torch/ST/pinecone + keys) ----------
    if not CONFIG.pinecone.api_key:
        sys.exit("PINECONE_API_KEY not set — cannot index. Use --dry-run to test parsing.")

    from biomed_rag.embeddings import SparseEncoder, get_dense_embed_model, save_encoder_meta
    from biomed_rag.index import PineconeHybridIndex, namespace_for

    os.makedirs(args.artifacts_dir, exist_ok=True)
    print(f"[ingest] loading dense model: {CONFIG.embed.model}")
    dense = get_dense_embed_model(CONFIG.embed.model)
    probe_dim = len(dense.get_text_embedding("dimension probe"))

    print("[ingest] fitting BM25 on corpus ...")
    texts = [n.get_content() for n in nodes]
    sparse = SparseEncoder().fit(texts)

    index = PineconeHybridIndex(CONFIG)
    index.ensure_index(dimension=probe_dim)
    ns = namespace_for(args.doc_type)
    print(f"[ingest] upserting {len(nodes)} nodes into namespace {ns!r} ...")
    n_up = index.upsert_nodes(nodes, dense, sparse, namespace=ns)

    sparse.save(os.path.join(args.artifacts_dir, "bm25.json"))
    save_encoder_meta(os.path.join(args.artifacts_dir, "embed_meta.json"),
                      CONFIG.embed.model, probe_dim)
    print(f"[ingest] upserted {n_up} vectors. Index stats:")
    print(json.dumps(index.stats(), indent=2, default=str))


if __name__ == "__main__":
    main()
