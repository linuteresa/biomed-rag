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
from ..trace import get_logger
from .dataset import Benchmark

log = get_logger(__name__)

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
                f"  top {k:<3d}  "
                f"recall={self.aggregate[f'recall@{k}']:.3f}  "
                f"precision={self.aggregate[f'precision@{k}']:.3f}  "
                f"f-score={self.aggregate[f'f1@{k}']:.3f}  "
                f"any-hit={self.aggregate[f'hit@{k}']:.3f}  "
                f"ranking-quality={self.aggregate[f'ndcg@{k}']:.3f}"
            )
        tail = (
            f"  first-hit-score={self.aggregate['mrr']:.3f}  "
            f"avg-precision={self.aggregate['map']:.3f}  "
            f"precision-at-N={self.aggregate['r_precision']:.3f}"
        )
        out = f"{self.name}  (n={self.n_queries})\n" + "\n".join(rows) + "\n" + tail
        if len(self.by_split) > 1:
            maxk = self.k_values[-1]
            splits = "  ".join(
                f"{s}(n={int(v['n'])}): recall@{maxk}={v[f'recall@{maxk}']:.3f} "
                f"ranking-quality@{maxk}={v[f'ndcg@{maxk}']:.3f} "
                f"first-hit-score={v['mrr']:.3f}"
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
    log.debug("evaluate(%s): %d queries, k=%s, pool=%d",
              name or benchmark.name, len(benchmark.queries), list(k_values), pool)

    for q in benchmark.queries:
        results = list(retrieve_fn(q.text, top_k=max(pool, max_k)))
        ranked = _ranked_pmids(results, id_of)
        rel: Mapping[str, float] = q.relevant
        first_hit = next((i for i, p in enumerate(ranked, 1) if p in q.relevant_pmids), None)
        log.debug("  [%s/%s] %r -> %d nodes / %d docs; first relevant at rank %s",
                  q.id, q.split, q.text, len(results), len(ranked), first_hit)

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

    log.info("evaluate(%s): recall@%d=%.3f ranking-quality@%d=%.3f first-hit-score=%.3f",
             name or benchmark.name, max_k, agg.get(f"recall@{max_k}", 0.0),
             max_k, agg.get(f"ndcg@{max_k}", 0.0), agg.get("mrr", 0.0))
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
    """Side-by-side table. If `baseline` names one of the reports, a
    "change vs baseline" column is appended for every metric.

    Column headers show readable metric names (`ranking-quality@10` rather than
    `ndcg@10`); pair the table with `metrics.legend_lines(metrics)` for a key.
    """
    run_names = list(reports)
    headers = [M.display_name(m) for m in metrics]
    col = max(12, max((len(h) for h in headers), default=12) + 2)
    width = max((len(x) for x in run_names), default=8) + 2
    has_base = bool(baseline and baseline in reports)

    header = "run".ljust(width) + "".join(h.rjust(col) for h in headers)
    if has_base:
        # "Δ" columns show the change versus the baseline run.
        header += "   " + "".join(f"Δ{h}".rjust(col) for h in headers)
    lines = [header, "-" * len(header)]
    base = reports[baseline].aggregate if has_base else None
    for name, rep in reports.items():
        cells = "".join(f"{rep.aggregate.get(m, 0.0):{col}.3f}" for m in metrics)
        line = name.ljust(width) + cells
        if base is not None:
            line += "   " + "".join(
                f"{rep.aggregate.get(m, 0.0) - base.get(m, 0.0):+{col}.3f}"
                for m in metrics
            )
        lines.append(line)
    return "\n".join(lines)
