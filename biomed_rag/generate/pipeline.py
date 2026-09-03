"""AnswerPipeline: retrieve -> prompt -> generate -> extract citations."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..trace import get_logger
from .llm import LLM, load_llm
from .prompt import build_messages, format_contexts

log = get_logger(__name__)

_PMID_CITE = re.compile(r"PMID:\s*([0-9]{5,9})")


@dataclass
class Answer:
    query: str
    text: str
    citations: list[str]            # PMIDs cited in the answer that are in-context
    uncited_claims: list[str]       # answer sentences with no in-context citation
    contexts: list = field(default_factory=list)          # RetrievalResult list
    context_pmids: list[str] = field(default_factory=list)

    @property
    def hallucinated_citations(self) -> list[str]:
        cited = set(_PMID_CITE.findall(self.text))
        return sorted(cited - set(self.context_pmids))

    def format(self) -> str:
        lines = [self.text, "", "Sources:"]
        seen = set()
        for r in self.contexts:
            md = getattr(r, "metadata", {}) or {}
            pmid = md.get("pmid", "?")
            if pmid in seen:
                continue
            seen.add(pmid)
            mark = "*" if pmid in self.citations else " "
            lines.append(
                f" [{mark}] PMID:{pmid}  {md.get('journal','')} {md.get('year','')}".rstrip()
            )
        return "\n".join(lines)


class AnswerPipeline:
    def __init__(
        self,
        retriever,
        llm: LLM,
        top_k: int = 5,
        alpha: float = 0.5,
        rerank: bool = False,
        max_new_tokens: int = 256,
    ):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.alpha = alpha
        self.rerank = rerank
        self.max_new_tokens = max_new_tokens

    @classmethod
    def build(
        cls,
        retriever,
        llm: Optional[LLM | str] = None,
        **kwargs,
    ) -> "AnswerPipeline":
        model = llm if isinstance(llm, LLM) else load_llm(llm)
        log.debug("AnswerPipeline.build: llm backend = %s", getattr(model, "name", model))
        return cls(retriever, model, **kwargs)

    def answer(
        self,
        query: str,
        top_k: Optional[int] = None,
        alpha: Optional[float] = None,
        rerank: Optional[bool] = None,
        namespace: Optional[str] = None,
        metadata_filter: Optional[dict] = None,
    ) -> Answer:
        k = top_k or self.top_k
        log.debug("AnswerPipeline.answer: %r (llm=%s, top_k=%d)", query, self.llm.name, k)
        results = list(
            self.retriever.retrieve(
                query,
                top_k=k,
                alpha=self.alpha if alpha is None else alpha,
                namespace=namespace,
                metadata_filter=metadata_filter,
                rerank=self.rerank if rerank is None else rerank,
            )
        )
        _ctx, ctx_pmids = format_contexts(results)
        log.debug("answer: %d context node(s), pmids=%s", len(results), ctx_pmids)
        messages = build_messages(query, results)
        text = self.llm.generate(messages, max_new_tokens=self.max_new_tokens).strip()
        log.debug("answer: generated %d char(s)", len(text))

        cited = set(_PMID_CITE.findall(text))
        in_context = [p for p in ctx_pmids if p in cited]

        from .llm import split_sentences

        uncited = [
            s for s in split_sentences(text)
            if not _PMID_CITE.search(s) and "do not contain enough information" not in s.lower()
        ]
        hallucinated = sorted(cited - set(ctx_pmids))
        log.debug("answer: %d in-context citation(s), %d hallucinated, %d uncited sentence(s)",
                  len(in_context), len(hallucinated), len(uncited))
        return Answer(
            query=query,
            text=text,
            citations=in_context,
            uncited_claims=uncited,
            contexts=results,
            context_pmids=ctx_pmids,
        )
