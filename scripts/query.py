#!/usr/bin/env python3
"""Query the biomed-rag hybrid index.

Examples
--------
python scripts/query.py "does IL-6 drive macrophage polarization?" --top-k 5
python scripts/query.py "SGLT2 heart failure" --alpha 0.3 --year-min 2020 --rerank
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biomed_rag.config import CONFIG
from biomed_rag.index import namespace_for
from biomed_rag.retrieve import build_metadata_filter, build_retriever


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("query")
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.5,
                   help="1.0=pure dense, 0.0=pure BM25 sparse")
    p.add_argument("--doc-type", default=CONFIG.DOC_TYPE_ABSTRACT)
    p.add_argument("--year-min", type=int)
    p.add_argument("--year-max", type=int)
    p.add_argument("--journal")
    p.add_argument("--mesh", help="require this MeSH descriptor")
    p.add_argument("--rerank", action="store_true", help="apply cross-encoder re-ranking")
    p.add_argument("--artifacts-dir", default="data/cache")
    args = p.parse_args()

    retriever = build_retriever(args.artifacts_dir, with_reranker=args.rerank)
    mfilter = build_metadata_filter(
        year_min=args.year_min, year_max=args.year_max,
        journal=args.journal, mesh=args.mesh, doc_type=args.doc_type,
    )
    results = retriever.retrieve(
        args.query, top_k=args.top_k, alpha=args.alpha,
        namespace=namespace_for(args.doc_type),
        metadata_filter=mfilter, rerank=args.rerank,
    )
    print(f"\nQuery: {args.query}")
    print(f"alpha={args.alpha}  top_k={args.top_k}  rerank={args.rerank}  filter={mfilter}\n")
    for i, r in enumerate(results, 1):
        m = r.metadata
        print(f"{i:2d}. [{r.score:.4f}] {m.get('pmid','?')} ({m.get('year','?')}) "
              f"{m.get('journal','')} — sec={m.get('section','-')}")
        print(f"     {r.text[:180].strip()}...")


if __name__ == "__main__":
    main()
