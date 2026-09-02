"""Retriever-agnostic evaluation loop.

`evaluate()` takes any callable with the signature

    retrieve_fn(query: str, top_k: int) -> list[RetrievalResult]

(the production `HybridRetriever.retrieve` and the `OfflineHybridRetriever`
both satisfy it once their extra knobs are bound with `functools.partial`),
runs it over a `Benchmark`, maps every retrieved node back to its source PMID,
de-duplicates to a document ranking, and scores that ranking with `metrics`.

`compare()` renders several reports side by side — that's how the alpha sweep
and the re-ranker ablation are read off.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping, Sequence

from . import metrics as M
from .dataset import Benchmark

RetrieveFn = Callable[..., Sequence]  # (query, top_k=...) -> [RetrievalResult]


def _result_pmid(r) -> str:
    md = getattr(r, "metadata", {}) or {}
    return md.get("pmid") or md.get("PMID") or ""


def _ranked_pmids(results: Sequence, id_of: Callable[[object], str]) -> list[str]:
    """Collapse a node ranking to a document (PMID) ranking, keeping first
    occurrence order and dropping blanks."""
    seen: set[str] = set()
    out: list[str] = []
    for r in results:
        pmid = id_of(r)
        if pmid and pmid not in seen:
            seen.add(pmid)
            out.append(pmid)
    return out


@dataclass
class EvalReport:
    name: str
    n_queries: int
    k_values: tuple[int, ...]
    aggregate: dict[str, float]
    per_query: list[dict] = field(default_factory=list)
    by_split: dict[str, dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "n_queries": self.n_queries,
            "k_values": list(self.k_values),
            "aggregate": self.aggregate,
            "by_split": self.by_split,
            "per_query": self.per_query,
        }

    def table(self) -> str:
        rows = []
        for k in self.k_values:
            rows.append(
                f"  k={k:<3d}  "
                f"recall={self.aggregate[f'recall@{k}']:.3f}  "
                f"precision={self.aggregate[f'precision@{k}']:.3f}  "
                f"f1={self.aggregate[f'f1@{k}']:.3f}  "
                f"hit={self.aggregate[f'hit@{k}']:.3f}  "
                f"ndcg={self.aggregate[f'ndcg@{k}']:.3f}"
            )
        tail = (
            f"  MRR={self.aggregate['mrr']:.3f}  "
            f"MAP={self.aggregate['map']:.3f}  "
            f"R-Precision={self.aggregate['r_precision']:.3f}"
        )
        out = f"{self.name}  (n={self.n_queries})\n" + "\n".join(rows) + "\n" + tail
        if len(self.by_split) > 1:
            maxk = self.k_values[-1]
            splits = "  ".join(
                f"{s}(n={int(v['n'])}): recall@{maxk}={v[f'recall@{maxk}']:.3f} "
                f"ndcg@{maxk}={v[f'ndcg@{maxk}']:.3f} mrr={v['mrr']:.3f}"
                for s, v in sorted(self.by_split.items())
            )
            out += "\n  by split -> " + splits
        return out


def evaluate(
    retrieve_fn: RetrieveFn,
    benchmark: Benchmark,
    k_values: Sequence[int] = (1, 3, 5, 10),
    pool: int = 100,
    id_of: Callable[[object], str] = _result_pmid,
    name: str | None = None,
) -> EvalReport:
    k_values = tuple(sorted(set(int(k) for k in k_values)))
    max_k = max(k_values)
    per_query: list[dict] = []

    for q in benchmark.queries:
        results = list(retrieve_fn(q.text, top_k=max(pool, max_k)))
        ranked = _ranked_pmids(results, id_of)
        rel: Mapping[str, float] = q.relevant

        row: dict[str, float | str | list] = {"query_id": q.id, "n_retrieved": len(ranked)}
        for k in k_values:
            for mname, fn in M.AT_K_METRICS.items():
                row[f"{mname}@{k}"] = fn(ranked, rel, k)
        for mname, fn in M.GLOBAL_METRICS.items():
            row[mname] = fn(ranked, rel)
        row["first_relevant_rank"] = next(
            (i for i, p in enumerate(ranked, 1) if p in q.relevant_pmids), None
        )
        row["split"] = q.split
        per_query.append(row)

    metric_keys = [f"{m}@{k}" for k in k_values for m in M.AT_K_METRICS] + list(
        M.GLOBAL_METRICS
    )

    def _agg(rows: list[dict]) -> dict[str, float]:
        return {key: M.mean(r[key] for r in rows) for key in metric_keys}

    agg = _agg(per_query)
    splits = sorted({r["split"] for r in per_query})
    by_split = {}
    if len(splits) > 1:
        for s in splits:
            rows = [r for r in per_query if r["split"] == s]
            by_split[s] = {**_agg(rows), "n": float(len(rows))}

    return EvalReport(
        name=name or benchmark.name,
        n_queries=len(benchmark.queries),
        k_values=k_values,
        aggregate=agg,
        per_query=per_query,
        by_split=by_split,
    )


def compare(
    reports: Mapping[str, EvalReport],
    metrics: Sequence[str] = ("recall@10", "ndcg@10", "mrr", "map"),
    baseline: str | None = None,
) -> str:
    """Side-by-side table. If `baseline` names one of the reports, a Δ vs that
    baseline is appended to every other row."""
    labels = list(reports)
    width = max((len(x) for x in labels), default=8) + 2
    header = "run".ljust(width) + "".join(m.rjust(12) for m in metrics)
    if baseline and baseline in reports:
        header += "   " + "  ".join(f"Δ{m}" for m in metrics)
    lines = [header, "-" * len(header)]
    base = reports[baseline].aggregate if (baseline and baseline in reports) else None
    for label, rep in reports.items():
        cells = "".join(f"{rep.aggregate.get(m, 0.0):12.3f}" for m in metrics)
        line = label.ljust(width) + cells
        if base is not None:
            deltas = "  ".join(
                f"{rep.aggregate.get(m, 0.0) - base.get(m, 0.0):+7.3f}" for m in metrics
            )
            line += "   " + deltas
        lines.append(line)
    return "\n".join(lines)
