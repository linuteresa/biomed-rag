#!/usr/bin/env python3
"""Evaluate retrieval (and, with --generation, grounded answering) on the
labelled benchmark in data/benchmarks/.

Runs fully offline by default: a pure-Python BM25 + TF-IDF hybrid retriever over
the synthetic corpus, no API keys, no model download.

Examples
--------
# Offline alpha sweep + re-ranker ablation, easy/hard split breakdown:
python scripts/eval.py

python scripts/eval.py --alphas 0,0.25,0.5,0.75,1.0 --k 1,3,5,10 --out data/benchmarks/results.json

# Add the grounded-generation grounding metrics (extractive LLM, still offline):
python scripts/eval.py --generation

# Evaluate the real Pinecone hybrid retriever instead (needs keys + a built index):
python scripts/eval.py --live --artifacts-dir data/cache --rerank
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from biomed_rag.eval import (
    compare,
    evaluate,
    load_benchmark,
    load_corpus,
    validate_benchmark,
)
from biomed_rag.eval.metrics import legend_lines as metrics_legend
from biomed_rag.eval.offline import LexicalReranker, OfflineHybridRetriever


def _floats(s: str) -> list[float]:
    return [float(x) for x in s.split(",") if x.strip() != ""]


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip() != ""]


def build_offline(corpus, with_reranker: bool):
    reranker = LexicalReranker() if with_reranker else None
    return OfflineHybridRetriever.from_corpus(corpus, reranker=reranker)


def build_live(artifacts_dir: str, with_reranker: bool):
    from biomed_rag.retrieve import build_retriever

    return build_retriever(artifacts_dir, with_reranker=with_reranker)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--alphas", type=_floats, default=[0.0, 0.25, 0.5, 0.75, 1.0],
                   help="comma-separated hybrid alphas to sweep (1=dense, 0=BM25)")
    p.add_argument("--k", dest="k_values", type=_ints, default=[1, 3, 5, 10])
    p.add_argument("--rerank", action="store_true",
                   help="also evaluate the best alpha with the re-ranker (ablation)")
    p.add_argument("--generation", action="store_true",
                   help="also compute grounded-answer metrics over the benchmark")
    p.add_argument("--live", action="store_true",
                   help="evaluate the real Pinecone hybrid retriever (needs keys + index)")
    p.add_argument("--artifacts-dir", default="data/cache")
    p.add_argument("--pool", type=int, default=100, help="candidate pool per query")
    p.add_argument("--out", help="write the full report bundle as JSON here")
    args = p.parse_args()

    bench = load_benchmark()
    corpus = load_corpus()
    problems = validate_benchmark(bench, corpus)
    for msg in problems:
        print(f"[validate] {msg}")
    print(f"[eval] benchmark={bench.name}  queries={len(bench)}  corpus_docs={len(corpus)}")

    mode = "live" if args.live else "offline"
    print(f"[eval] mode={mode}  metric k-values={args.k_values}\n")

    reports = {}
    for a in args.alphas:
        retr = build_live(args.artifacts_dir, False) if args.live else build_offline(corpus, False)
        rep = evaluate(partial(retr.retrieve, alpha=a), bench,
                       k_values=args.k_values, pool=args.pool, name=f"alpha={a:g}")
        reports[rep.name] = rep

    baseline = min(reports, key=lambda n: reports[n].aggregate[f"ndcg@{max(args.k_values)}"])
    maxk = max(args.k_values)
    sweep_metrics = (f"recall@{maxk}", f"ndcg@{maxk}", "mrr", "map")
    ablation_metrics = ("precision@1", f"ndcg@{maxk}", "mrr", "map")

    # One legend up front covering every column that appears below: the alpha
    # sweep uses `sweep_metrics`, the re-ranker ablation adds precision@1.
    print("How to read the columns:")
    for line in metrics_legend(sweep_metrics + ablation_metrics):
        print(f"  {line}")
    print("  alpha ...........  1.0 = dense only, 0.0 = keyword (BM25) only\n")

    print("=== alpha sweep " + "=" * 50)
    print(compare(reports, metrics=sweep_metrics, baseline=baseline))

    best_name = max(reports, key=lambda n: reports[n].aggregate[f"ndcg@{maxk}"])
    best_alpha = float(best_name.split("=")[1])
    print(f"\n[eval] best alpha by ranking-quality (top {maxk}): {best_alpha:g}")

    if any(r.by_split for r in reports.values()):
        print("\n=== split breakdown (best alpha) " + "=" * 34)
        bs = reports[best_name].by_split
        for s, v in sorted(bs.items()):
            print(f"  {s:<6} n={int(v['n']):<3} "
                  + "  ".join(f"recall@{k}={v[f'recall@{k}']:.3f}" for k in args.k_values)
                  + f"  ranking-quality@{maxk}={v[f'ndcg@{maxk}']:.3f}"
                  + f"  first-hit-score={v['mrr']:.3f}")

    ablation = {}
    if args.rerank:
        print("\n=== re-ranker ablation (alpha=" + f"{best_alpha:g}) " + "=" * 33)
        retr = build_live(args.artifacts_dir, True) if args.live else build_offline(corpus, True)
        no_rr = evaluate(partial(retr.retrieve, alpha=best_alpha), bench,
                         k_values=args.k_values, pool=args.pool, name="no-rerank")
        with_rr = evaluate(partial(retr.retrieve, alpha=best_alpha, rerank=True), bench,
                           k_values=args.k_values, pool=args.pool, name="rerank")
        ablation = {"no-rerank": no_rr, "rerank": with_rr}
        print(compare(ablation, metrics=ablation_metrics, baseline="no-rerank"))
        if mode == "offline":
            print("  note: offline uses a model-free LexicalReranker; the production "
                  "cross-encoder (retrieve.CrossEncoderReranker) needs a model download.")

    gen_report = None
    if args.generation:
        print("\n=== grounded generation " + "=" * 42)
        from biomed_rag.generate import AnswerPipeline
        from biomed_rag.generate.evaluate import evaluate_generation

        retr = build_live(args.artifacts_dir, args.rerank) if args.live else build_offline(corpus, args.rerank)
        pipe = AnswerPipeline.build(retriever=retr)  # extractive LLM unless a model is set
        gen_report = evaluate_generation(pipe, bench, alpha=best_alpha, rerank=args.rerank)
        gen_labels = {
            "faithfulness": "faithfulness ....... answer stays grounded in the sources",
            "answer_relevancy": "answer-relevancy ... answer is on-topic for the question",
            "citation_support": "citation-support ... answer sentences carry a source tag",
            "hallucinated_citation": "made-up-citations .. cites a source that was not retrieved",
            "context_precision": "context-precision . share of retrieved docs that are on target",
            "context_recall": "context-recall .... share of on-target docs that were retrieved",
            "abstention_rate": "abstention-rate ... share of questions answered \"not enough info\"",
        }
        for key, val in gen_report["aggregate"].items():
            print(f"  {gen_labels.get(key, key):<58} {val:.3f}")
        print(f"  [answer generator: {gen_report['llm_backend']}]")

    if args.out:
        bundle = {
            "benchmark": bench.name,
            "mode": mode,
            "corpus_docs": len(corpus),
            "alpha_sweep": {n: r.to_dict() for n, r in reports.items()},
            "best_alpha": best_alpha,
            "rerank_ablation": {n: r.to_dict() for n, r in ablation.items()},
            "generation": gen_report,
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(bundle, fh, indent=2, default=str)
        print(f"\n[eval] wrote {args.out}")


if __name__ == "__main__":
    main()
