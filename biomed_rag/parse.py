"""LlamaIndex parsing layer: BiomedDocuments -> Documents -> Nodes.

Two node-parsing strategies, matched to doc_type:

* abstracts  -> section-aware chunking. A structured abstract's BACKGROUND /
               METHODS / RESULTS / CONCLUSIONS blocks each become their own
               node (further split if long), so a query about *results* isn't
               diluted by the *methods* text sharing one chunk. Each node keeps
               a `section` metadata field for filtering / display.
* full text  -> HierarchicalNodeParser (2048 / 512 / 128), giving parent/child
               nodes for auto-merging retrieval of long PMC bodies.

Metadata from the document is propagated onto every node automatically by
LlamaIndex, so downstream metadata filtering (year, journal, mesh_terms,
doc_type, section) works on nodes without extra bookkeeping.
"""
from __future__ import annotations

from typing import Sequence

from llama_index.core import Document
from llama_index.core.node_parser import HierarchicalNodeParser, SentenceSplitter
from llama_index.core.schema import BaseNode, TextNode

from .config import CONFIG
from .ingest import BiomedDocument

# Keep identifiers and long lists out of the embedded/LLM text, but let the
# semantically useful fields (title, journal, year, MeSH, section) enrich it.
_EXCLUDE_EMBED = ["pmid", "doi", "pmcid", "source", "authors", "keywords", "doc_type"]
_EXCLUDE_LLM = ["pmid", "doi", "pmcid", "source", "authors", "doc_type"]


def to_llama_documents(docs: Sequence[BiomedDocument]) -> list[Document]:
    """Wrap BiomedDocuments as LlamaIndex Documents (one per record)."""
    out: list[Document] = []
    for d in docs:
        out.append(
            Document(
                doc_id=d.doc_id,
                text=d.text,
                metadata=dict(d.metadata),
                excluded_embed_metadata_keys=list(_EXCLUDE_EMBED),
                excluded_llm_metadata_keys=list(_EXCLUDE_LLM),
                metadata_seperator="\n",
                metadata_template="{key}: {value}",
                text_template="{metadata_str}\n\n{content}",
            )
        )
    return out


def _section_documents(d: BiomedDocument) -> list[Document]:
    """Explode a structured abstract into one Document per labelled section.

    The title is prepended to each section so an isolated node still carries
    the topic context that dense retrieval relies on.
    """
    base_meta = dict(d.metadata)
    docs: list[Document] = []
    for i, sec in enumerate(d.abstract_sections):
        label = sec.get("label") or "ABSTRACT"
        text = sec.get("text", "").strip()
        if not text:
            continue
        meta = {**base_meta, "section": label}
        docs.append(
            Document(
                doc_id=f"{d.doc_id}::sec{i}:{label}",
                text=f"{d.title}\n\n{text}",
                metadata=meta,
                excluded_embed_metadata_keys=list(_EXCLUDE_EMBED),
                excluded_llm_metadata_keys=list(_EXCLUDE_LLM),
                metadata_seperator="\n",
                metadata_template="{key}: {value}",
                text_template="{metadata_str}\n\n{content}",
            )
        )
    return docs


def build_nodes(
    docs: Sequence[BiomedDocument],
    abstract_chunk_size: int = 256,
    abstract_chunk_overlap: int = 32,
) -> list[BaseNode]:
    """Turn BiomedDocuments into retrievable nodes using the right strategy."""
    abstract_splitter = SentenceSplitter(
        chunk_size=abstract_chunk_size, chunk_overlap=abstract_chunk_overlap
    )
    hierarchical = HierarchicalNodeParser.from_defaults(chunk_sizes=[2048, 512, 128])

    nodes: list[BaseNode] = []
    for d in docs:
        if d.doc_type == CONFIG.DOC_TYPE_FULLTEXT:
            # Long body: hierarchical parent/child nodes for auto-merging.
            doc = to_llama_documents([d])[0]
            nodes.extend(hierarchical.get_nodes_from_documents([doc]))
        elif len(d.abstract_sections) > 1:
            # Structured abstract: section-aware nodes, each split if long.
            section_docs = _section_documents(d)
            nodes.extend(abstract_splitter.get_nodes_from_documents(section_docs))
        else:
            # Unstructured abstract / short doc: plain sentence splitting.
            doc = to_llama_documents([d])[0]
            nodes.extend(abstract_splitter.get_nodes_from_documents([doc]))
    return nodes


def node_summary(nodes: Sequence[BaseNode]) -> dict:
    """Small stats helper used by the dry-run path."""
    by_section: dict[str, int] = {}
    by_doc: dict[str, int] = {}
    for n in nodes:
        sec = n.metadata.get("section", "-")
        by_section[sec] = by_section.get(sec, 0) + 1
        pmid = n.metadata.get("pmid", "-")
        by_doc[pmid] = by_doc.get(pmid, 0) + 1
    lengths = [len(getattr(n, "text", "") or "") for n in nodes]
    return {
        "n_nodes": len(nodes),
        "by_section": by_section,
        "nodes_per_doc": by_doc,
        "avg_chars": round(sum(lengths) / len(lengths), 1) if lengths else 0,
    }
