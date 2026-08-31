"""Offline smoke tests for the ingestion layer.

These exercise the full PubMed-XML -> BiomedDocument path with only the Python
standard library — no network, no API keys, no model downloads — so `pytest`
runs anywhere the repo is checked out. The framework layers (parse/index/
retrieve) are covered by higher-level integration tests that require the ML
dependencies and live services.
"""
import os

import pytest

from biomed_rag.ingest import (
    BiomedDocument,
    load_records_from_file,
    parse_pubmed_xml,
    records_to_documents,
)

FIXTURE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "sample", "pubmed_sample.xml",
)


@pytest.fixture(scope="module")
def records():
    return load_records_from_file(FIXTURE)


@pytest.fixture(scope="module")
def docs(records):
    return records_to_documents(records, doc_type="abstract")


def test_parses_all_records(records):
    assert len(records) == 4
    assert {r["pmid"] for r in records} == {
        "90000001", "90000002", "90000003", "90000004",
    }


def test_structured_abstract_sections(records):
    r = next(r for r in records if r["pmid"] == "90000001")
    labels = [s["label"] for s in r["abstract_sections"]]
    assert labels == ["BACKGROUND", "METHODS", "RESULTS", "CONCLUSIONS"]


def test_medline_date_year_parsing(records):
    # Record 3's PubDate is a MedlineDate "2020 Dec-2021 Jan" -> first year 2020.
    r = next(r for r in records if r["pmid"] == "90000003")
    assert r["year"] == 2020


def test_collective_author_and_elocation_doi(records):
    r = next(r for r in records if r["pmid"] == "90000002")
    assert "Synthetic Lung Cancer Consortium" in r["authors"]
    # DOI provided only via ELocationID, not ArticleIdList.
    assert r["doi"] == "10.9999/synth.2019.0142"


def test_mesh_terms_extracted(records):
    r = next(r for r in records if r["pmid"] == "90000004")
    assert "Glymphatic System" in r["mesh_terms"]
    assert "Amyloid beta-Peptides" in r["mesh_terms"]


def test_document_metadata_is_pinecone_safe(docs):
    allowed = (str, int, float, bool)
    for d in docs:
        assert isinstance(d, BiomedDocument)
        assert d.doc_id.startswith("PMID:")
        for key, val in d.metadata.items():
            if isinstance(val, list):
                assert all(isinstance(x, str) for x in val), key
            else:
                assert isinstance(val, allowed), f"{key}={val!r}"
        # No empty values should survive cleaning.
        assert "" not in d.metadata.values()


def test_document_text_includes_title_and_body(docs):
    d = next(d for d in docs if d.metadata["pmid"] == "90000001")
    assert d.text.startswith("Interleukin-6 signaling")
    assert "BACKGROUND:" in d.text
    assert "STAT3" in d.text


def test_empty_xml_is_safe():
    assert parse_pubmed_xml(b"<PubmedArticleSet/>") == []
