#!/usr/bin/env python3
"""Answer a question with grounded, cited generation over retrieved nodes.

Offline by default: BM25 + TF-IDF hybrid retrieval over the synthetic benchmark
corpus, plus the deterministic ExtractiveLLM. Point --model at a cached
HuggingFace instruct checkpoint for abstractive synthesis, or --live at the
real Pinecone index.

Examples
--------
python scripts/answer.py "does IL-6 drive M2 macrophage polarization via STAT3?"
python scripts/answer.py "how is amyloid cleared from the brain during sleep?" --top-k 4 --alpha 0.3
python scripts/answer.py "semaglutide vs tirzepatide for weight loss" --model HuggingFaceTB/SmolLM2-135M-Instruct
python scripts/answer.py "carbapenem-resistant Klebsiella treatment" --live --artifacts-dir data/cache --rerank
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biomed_rag.generate import AnswerPipeline
from biomed_rag.generate.evaluate import faithfulness


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("query")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--rerank", action="store_true")
    p.add_argument("--model", default="extractive",
                   help="'extractive' (default) or a HuggingFace model id/path")
    p.add_argument("--live", action="store_true",
                   help="retrieve from the real Pinecone index instead of offline")
    p.add_argument("--artifacts-dir", default="data/cache")
    args = p.parse_args()

    if args.live:
        from biomed_rag.retrieve import build_retriever

        retriever = build_retriever(args.artifacts_dir, with_reranker=args.rerank)
    else:
        from biomed_rag.eval import load_corpus
        from biomed_rag.eval.offline import LexicalReranker, OfflineHybridRetriever

        retriever = OfflineHybridRetriever.from_corpus(
            load_corpus(), reranker=LexicalReranker() if args.rerank else None
        )

    pipe = AnswerPipeline.build(
        retriever=retriever, llm=args.model,
        top_k=args.top_k, alpha=args.alpha, rerank=args.rerank,
    )
    ans = pipe.answer(args.query)

    print(f"\nQ: {args.query}")
    print(f"[retriever={'live' if args.live else 'offline'}  llm={pipe.llm.name}  "
          f"alpha={args.alpha}  top_k={args.top_k}  rerank={args.rerank}]\n")
    print(ans.format())
    print(f"\nfaithfulness (heuristic): {faithfulness(ans.text, ans.contexts):.2f}")
    if ans.hallucinated_citations:
        print(f"WARNING: cited PMIDs not in context: {ans.hallucinated_citations}")
    if ans.uncited_claims:
        print(f"uncited sentences: {len(ans.uncited_claims)}")


if __name__ == "__main__":
    main()
