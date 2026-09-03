"""Reference-free grounding metrics for generated answers.

RAGAS-style *in spirit* — faithfulness, answer relevancy, context
precision/recall — but computed with token-overlap heuristics instead of an
LLM judge, so the whole thing runs offline and deterministically. Treat the
numbers as directional proxies, not the published RAGAS metrics.

* faithfulness      — fraction of answer sentences whose content is supported
                      (high token containment) by at least one retrieved passage.
* answer_relevancy  — token cosine between the answer and the question.
* citation_support  — fraction of answer sentences carrying an in-context
                      [PMID:...] citation (abstention counts as supported).
* hallucinated_cite — fraction of answers citing a PMID that was not retrieved.
* context_precision — fraction of retrieved documents that are gold-relevant.
* context_recall    — fraction of gold-relevant documents that were retrieved.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import Sequence

from ..trace import get_logger
from .llm import split_sentences

log = get_logger(__name__)

_WORD_RE = re.compile(r"[a-z0-9][a-z0-9\-]*")
_PMID_CITE = re.compile(r"PMID:\s*([0-9]{5,9})")
_ABSTAIN = "do not contain enough information"


def _toks(text: str) -> list[str]:
    return [w for w in _WORD_RE.findall((text or "").lower()) if len(w) > 2]


def _containment(a: str, b: str) -> float:
    """Fraction of the content tokens of `a` that also appear in `b`."""
    ta, tb = set(_toks(a)), set(_toks(b))
    if not ta:
        return 1.0
    return len(ta & tb) / len(ta)


def _cosine(a: str, b: str) -> float:
    ca, cb = Counter(_toks(a)), Counter(_toks(b))
    if not ca or not cb:
        return 0.0
    dot = sum(ca[t] * cb.get(t, 0) for t in ca)
    na = math.sqrt(sum(v * v for v in ca.values()))
    nb = math.sqrt(sum(v * v for v in cb.values()))
    return dot / (na * nb) if na and nb else 0.0


def faithfulness(answer_text: str, contexts: Sequence, threshold: float = 0.6) -> float:
    if _ABSTAIN in (answer_text or "").lower():
        return 1.0
    sents = split_sentences(answer_text)
    if not sents:
        return 0.0
    ctx_texts = [getattr(c, "text", "") or "" for c in contexts]
    supported = 0
    for s in sents:
        s_clean = _PMID_CITE.sub("", s)
        if any(_containment(s_clean, ct) >= threshold for ct in ctx_texts):
            supported += 1
    return supported / len(sents)


def answer_relevancy(answer_text: str, query: str) -> float:
    if _ABSTAIN in (answer_text or "").lower():
        return 0.0
    return _cosine(_PMID_CITE.sub("", answer_text), query)


def citation_support(answer_text: str, context_pmids: Sequence[str]) -> float:
    if _ABSTAIN in (answer_text or "").lower():
        return 1.0
    sents = split_sentences(answer_text)
    if not sents:
        return 0.0
    ctx = set(context_pmids)
    ok = sum(1 for s in sents if any(p in ctx for p in _PMID_CITE.findall(s)))
    return ok / len(sents)


def has_hallucinated_citation(answer_text: str, context_pmids: Sequence[str]) -> bool:
    cited = set(_PMID_CITE.findall(answer_text or ""))
    return bool(cited - set(context_pmids))


def context_precision(context_pmids: Sequence[str], relevant_pmids: set[str]) -> float:
    uniq = list(dict.fromkeys(context_pmids))
    if not uniq:
        return 0.0
    return sum(1 for p in uniq if p in relevant_pmids) / len(uniq)


def context_recall(context_pmids: Sequence[str], relevant_pmids: set[str]) -> float:
    if not relevant_pmids:
        return 0.0
    ctx = set(context_pmids)
    return sum(1 for p in relevant_pmids if p in ctx) / len(relevant_pmids)


def evaluate_generation(
    pipeline,
    benchmark,
    top_k: int = 5,
    alpha: float | None = None,
    rerank: bool = False,
) -> dict:
    """Run the pipeline over every benchmark query and aggregate grounding
    metrics. Returns {aggregate, per_query, llm_backend}."""
    per_query: list[dict] = []
    log.debug("evaluate_generation: %d query/answer pair(s), top_k=%d alpha=%s rerank=%s",
              len(benchmark.queries), top_k, alpha, rerank)
    for q in benchmark.queries:
        ans = pipeline.answer(q.text, top_k=top_k, alpha=alpha, rerank=rerank)
        row = {
            "query_id": q.id,
            "split": q.split,
            "abstained": _ABSTAIN in ans.text.lower(),
            "faithfulness": faithfulness(ans.text, ans.contexts),
            "answer_relevancy": answer_relevancy(ans.text, q.text),
            "citation_support": citation_support(ans.text, ans.context_pmids),
            "hallucinated_citation": float(
                has_hallucinated_citation(ans.text, ans.context_pmids)
            ),
            "context_precision": context_precision(ans.context_pmids, q.relevant_pmids),
            "context_recall": context_recall(ans.context_pmids, q.relevant_pmids),
        }
        log.debug("  [%s/%s] abstained=%s faithfulness=%.2f ctx-precision=%.2f ctx-recall=%.2f",
                  q.id, q.split, row["abstained"], row["faithfulness"],
                  row["context_precision"], row["context_recall"])
        per_query.append(row)

    keys = [
        "faithfulness",
        "answer_relevancy",
        "citation_support",
        "hallucinated_citation",
        "context_precision",
        "context_recall",
    ]
    n = len(per_query) or 1
    aggregate = {k: sum(r[k] for r in per_query) / n for k in keys}
    aggregate["abstention_rate"] = sum(r["abstained"] for r in per_query) / n
    return {
        "aggregate": aggregate,
        "per_query": per_query,
        "llm_backend": getattr(pipeline.llm, "name", "unknown"),
    }
