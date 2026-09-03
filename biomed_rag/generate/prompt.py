"""Prompt construction for grounded answering.

The context block numbers every retrieved passage and stamps it with its PMID,
journal and year, so the model can cite with an inline ``[PMID:12345678]`` tag
and a reader can trace every claim. The system prompt pins the model to the
supplied context and asks it to abstain when the answer is not there.
"""
from __future__ import annotations

from typing import Sequence

from ..trace import get_logger

log = get_logger(__name__)

SYSTEM_PROMPT = (
    "You are a biomedical research assistant. Answer the question using ONLY the "
    "numbered sources provided. Cite every claim with an inline tag of the form "
    "[PMID:XXXXXXXX] naming the source it came from. If the sources do not "
    "contain the answer, reply exactly: 'The retrieved sources do not contain "
    "enough information to answer this question.' Be concise (2-4 sentences) and "
    "do not add outside knowledge."
)

_MAX_CTX_CHARS = 600


def format_contexts(results: Sequence) -> tuple[str, list[str]]:
    """Render retrieved results as a numbered source list.

    Returns (context_text, ordered_unique_pmids).
    """
    lines: list[str] = []
    pmids: list[str] = []
    for i, r in enumerate(results, start=1):
        md = getattr(r, "metadata", {}) or {}
        pmid = md.get("pmid") or md.get("PMID") or "?"
        if pmid not in pmids:
            pmids.append(pmid)
        journal = md.get("journal", "")
        year = md.get("year", "")
        section = md.get("section")
        tag = f"PMID:{pmid}"
        head = f"[S{i}] ({tag}"
        if journal or year:
            head += f", {journal} {year}".rstrip()
        if section:
            head += f", section: {section}"
        head += ")"
        text = (getattr(r, "text", "") or "").strip().replace("\n", " ")
        if len(text) > _MAX_CTX_CHARS:
            text = text[:_MAX_CTX_CHARS].rsplit(" ", 1)[0] + " ..."
        lines.append(f"{head}\n{text}")
    log.debug("format_contexts: %d passage(s) -> %d unique pmid(s)", len(lines), len(pmids))
    return "\n\n".join(lines), pmids


def build_messages(query: str, results: Sequence) -> list[dict]:
    context, _pmids = format_contexts(results)
    user = (
        f"Question: {query}\n\n"
        f"Sources:\n{context}\n\n"
        f"Answer (with [PMID:...] citations):"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
