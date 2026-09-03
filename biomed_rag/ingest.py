"""PubMed ingestion: NCBI E-utilities client + PubMed-XML parsing.

Design note: this module depends only on the Python standard library. The heavy
frameworks (LlamaIndex, Pinecone, sentence-transformers) live in the layers
above it (`parse`, `index`). Keeping the domain/ingestion layer framework-free
means the parser and document builder can be unit-tested with no network, no API
keys, and no model downloads — see tests/test_ingest.py.
"""
from __future__ import annotations

import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Iterable

from .config import CONFIG, Config
from .trace import get_logger

log = get_logger(__name__)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
_YEAR_RE = re.compile(r"(\d{4})")


# --------------------------------------------------------------------------- #
# Domain object
# --------------------------------------------------------------------------- #
@dataclass
class BiomedDocument:
    """A single ingested record, independent of any RAG framework.

    `metadata` holds only Pinecone-safe values (str, int, float, bool, or
    list[str]) so it can be attached to vectors as-is. `abstract_sections`
    preserves the structured-abstract labels (BACKGROUND / METHODS / RESULTS ...)
    so the parsing layer can build section-aware nodes.
    """

    doc_id: str
    title: str
    abstract_sections: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    doc_type: str = "abstract"

    @property
    def abstract(self) -> str:
        parts = []
        for sec in self.abstract_sections:
            label, text = sec.get("label"), sec.get("text", "")
            parts.append(f"{label}: {text}" if label else text)
        return "\n".join(p for p in parts if p.strip())

    @property
    def text(self) -> str:
        """Full document text: title followed by the (structured) abstract."""
        body = self.abstract
        return f"{self.title}\n\n{body}".strip() if body else self.title.strip()


# --------------------------------------------------------------------------- #
# XML parsing (pure stdlib)
# --------------------------------------------------------------------------- #
def _text(el: ET.Element | None) -> str:
    """Flatten an element's text, including tail text of inline children
    (PubMed abstracts embed <i>/<sup> etc. inside AbstractText)."""
    if el is None:
        return ""
    return re.sub(r"\s+", " ", "".join(el.itertext())).strip()


def _first_year(el: ET.Element | None) -> int | None:
    if el is None:
        return None
    year_el = el.find("Year")
    raw = _text(year_el) if year_el is not None else _text(el.find("MedlineDate"))
    m = _YEAR_RE.search(raw or "")
    return int(m.group(1)) if m else None


def _authors(article: ET.Element) -> list[str]:
    out: list[str] = []
    for a in article.findall("./AuthorList/Author"):
        last = _text(a.find("LastName"))
        fore = _text(a.find("ForeName")) or _text(a.find("Initials"))
        collective = _text(a.find("CollectiveName"))
        if last:
            out.append(f"{last} {fore}".strip())
        elif collective:
            out.append(collective)
    return out


def _abstract_sections(article: ET.Element) -> list[dict]:
    sections: list[dict] = []
    for node in article.findall("./Abstract/AbstractText"):
        text = _text(node)
        if not text:
            continue
        label = (node.get("Label") or node.get("NlmCategory") or "").strip() or None
        sections.append({"label": label, "text": text})
    return sections


def _mesh_terms(medline: ET.Element) -> list[str]:
    return [
        _text(d)
        for d in medline.findall("./MeshHeadingList/MeshHeading/DescriptorName")
        if _text(d)
    ]


def _keywords(medline: ET.Element) -> list[str]:
    return [_text(k) for k in medline.findall("./KeywordList/Keyword") if _text(k)]


def _article_ids(pubmed_article: ET.Element) -> dict:
    ids = {"doi": "", "pmcid": ""}
    for aid in pubmed_article.findall("./PubmedData/ArticleIdList/ArticleId"):
        kind = (aid.get("IdType") or "").lower()
        val = _text(aid)
        if kind == "doi" and val:
            ids["doi"] = val
        elif kind == "pmc" and val:
            ids["pmcid"] = val
    # DOI is sometimes only in ELocationID
    if not ids["doi"]:
        for el in pubmed_article.findall(".//ELocationID"):
            if (el.get("EIdType") or "").lower() == "doi":
                ids["doi"] = _text(el)
                break
    return ids


def parse_pubmed_xml(xml_bytes: bytes | str) -> list[dict]:
    """Parse an efetch PubmedArticleSet into a list of plain record dicts.

    Returns dicts (not BiomedDocuments) so the parsing boundary stays simple and
    serialisable; call `records_to_documents` to build documents.
    """
    if isinstance(xml_bytes, str):
        xml_bytes = xml_bytes.encode("utf-8")
    root = ET.fromstring(xml_bytes)
    log.debug("parse_pubmed_xml: %d bytes, root=<%s>", len(xml_bytes), root.tag)

    records: list[dict] = []
    for pa in root.findall(".//PubmedArticle"):
        medline = pa.find("./MedlineCitation")
        if medline is None:
            continue
        article = medline.find("./Article")
        if article is None:
            continue

        pmid = _text(medline.find("./PMID"))
        journal = article.find("./Journal")
        journal_title = ""
        pub_year = None
        if journal is not None:
            journal_title = _text(journal.find("./ISOAbbreviation")) or _text(
                journal.find("./Title")
            )
            pub_year = _first_year(journal.find("./JournalIssue/PubDate"))
        ids = _article_ids(pa)

        records.append(
            {
                "pmid": pmid,
                "title": _text(article.find("./ArticleTitle")),
                "abstract_sections": _abstract_sections(article),
                "journal": journal_title,
                "year": pub_year,
                "authors": _authors(article),
                "mesh_terms": _mesh_terms(medline),
                "keywords": _keywords(medline),
                "doi": ids["doi"],
                "pmcid": ids["pmcid"],
            }
        )
    log.debug(
        "parse_pubmed_xml: %d PubmedArticle -> %d records",
        len(root.findall(".//PubmedArticle")),
        len(records),
    )
    return records


def _clean_metadata(md: dict) -> dict:
    """Drop empty values so Pinecone metadata stays compact and filterable."""
    out = {}
    for k, v in md.items():
        if v in (None, "", [], {}):
            continue
        out[k] = v
    return out


def records_to_documents(
    records: Iterable[dict], doc_type: str = "abstract"
) -> list[BiomedDocument]:
    docs: list[BiomedDocument] = []
    for r in records:
        pmid = r.get("pmid") or ""
        log.debug(
            "  doc pmid=%s year=%s sections=%d mesh=%d",
            pmid or "?",
            r.get("year"),
            len(r.get("abstract_sections", [])),
            len(r.get("mesh_terms", [])),
        )
        metadata = _clean_metadata(
            {
                "pmid": pmid,
                "title": r.get("title", ""),
                "journal": r.get("journal", ""),
                "year": r.get("year"),
                "authors": r.get("authors", []),
                "mesh_terms": r.get("mesh_terms", []),
                "keywords": r.get("keywords", []),
                "doi": r.get("doi", ""),
                "pmcid": r.get("pmcid", ""),
                "doc_type": doc_type,
                "source": "pubmed",
            }
        )
        docs.append(
            BiomedDocument(
                doc_id=f"PMID:{pmid}" if pmid else f"doc:{len(docs)}",
                title=r.get("title", ""),
                abstract_sections=r.get("abstract_sections", []),
                metadata=metadata,
                doc_type=doc_type,
            )
        )
    log.debug("records_to_documents: built %d documents (doc_type=%s)", len(docs), doc_type)
    return docs


# --------------------------------------------------------------------------- #
# NCBI E-utilities client
# --------------------------------------------------------------------------- #
class PubMedClient:
    """Minimal, rate-limited NCBI E-utilities client (esearch + efetch)."""

    def __init__(self, config: Config = CONFIG):
        self.cfg = config.ncbi
        self._min_interval = 1.0 / max(1, self.cfg.rate_limit_per_sec)
        self._last_call = 0.0

    def _base_params(self) -> dict:
        params = {"tool": self.cfg.tool}
        if self.cfg.email:
            params["email"] = self.cfg.email
        if self.cfg.api_key:
            params["api_key"] = self.cfg.api_key
        return params

    def _throttle(self) -> None:
        wait = self._min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def _request(self, endpoint: str, params: dict, retries: int = 3) -> bytes:
        url = f"{EUTILS}/{endpoint}"
        data = urllib.parse.urlencode(params).encode("utf-8")
        safe = {k: v for k, v in params.items() if k != "api_key"}
        last_err: Exception | None = None
        for attempt in range(retries):
            self._throttle()
            log.debug("NCBI %s attempt %d params=%s", endpoint, attempt + 1, safe)
            try:
                req = urllib.request.Request(
                    url, data=data, headers={"User-Agent": self.cfg.tool}
                )
                with urllib.request.urlopen(req, timeout=30) as resp:
                    body = resp.read()
                    log.debug("NCBI %s -> %d bytes (HTTP %s)", endpoint, len(body),
                              getattr(resp, "status", "?"))
                    return body
            except Exception as e:  # network / HTTP error -> backoff and retry
                last_err = e
                log.warning("NCBI %s attempt %d failed: %s", endpoint, attempt + 1, e)
                time.sleep(2**attempt)
        raise RuntimeError(f"NCBI {endpoint} failed after {retries} tries: {last_err}")

    def esearch(self, query: str, retmax: int = 50) -> list[str]:
        params = {
            **self._base_params(),
            "db": "pubmed",
            "term": query,
            "retmax": str(retmax),
            "retmode": "json",
        }
        import json

        payload = json.loads(self._request("esearch.fcgi", params))
        idlist = payload.get("esearchresult", {}).get("idlist", [])
        log.info("esearch %r (retmax=%d) -> %d pmids", query, retmax, len(idlist))
        return idlist

    def efetch(self, pmids: list[str]) -> bytes:
        log.debug("efetch: %d pmids", len(pmids))
        if not pmids:
            return b"<PubmedArticleSet/>"
        params = {
            **self._base_params(),
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        }
        return self._request("efetch.fcgi", params)

    def search_and_fetch(self, query: str, retmax: int = 50) -> list[dict]:
        pmids = self.esearch(query, retmax=retmax)
        xml = self.efetch(pmids)
        return parse_pubmed_xml(xml)


# --------------------------------------------------------------------------- #
# Offline helpers
# --------------------------------------------------------------------------- #
def load_records_from_file(path: str) -> list[dict]:
    """Parse a local PubMed-XML file (e.g. the bundled sample fixture)."""
    log.debug("load_records_from_file: %s", path)
    with open(path, "rb") as fh:
        return parse_pubmed_xml(fh.read())
