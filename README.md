# biomed-rag

A domain-expert RAG pipeline over **biomedical literature** (PubMed / PMC).
This repo is **Component 1** of a larger system: *ingestion + hybrid indexing +
retrieval*. It's built so the later components — a fine-tuned bi-encoder,
a measured cross-encoder re-ranker, a LoRA-tuned generator, and a RAGAS eval
harness — drop into seams that already exist here.

> Resume framing: *"Built a production RAG pipeline (LlamaIndex + Pinecone) with
> hybrid dense+BM25 retrieval, section-aware biomedical chunking, and a
> fine-tuned bi-encoder retriever; improved recall@10 by X% and cut hallucination
> rate by Y%, validated on a custom RAGAS harness."*

## What's built (Component 1)

| Module | Responsibility |
|---|---|
| `biomed_rag/config.py` | Env/`.env`-driven config (NCBI, Pinecone, embedding model). No side effects at import. |
| `biomed_rag/ingest.py` | **Stdlib-only** NCBI E-utilities client (esearch/efetch, rate-limited, retrying) + PubMed-XML parser + `BiomedDocument` builder. |
| `biomed_rag/parse.py` | LlamaIndex layer: `BiomedDocument → Document → Node`. Section-aware chunking for abstracts, `HierarchicalNodeParser` for full text. |
| `biomed_rag/embeddings.py` | Pluggable dense bi-encoder + BM25 sparse encoder + hybrid alpha-scaling. |
| `biomed_rag/index.py` | Pinecone hybrid store: serverless `dotproduct` index, per-doc-type namespaces, sparse-dense upsert, metadata filtering. |
| `biomed_rag/retrieve.py` | `HybridRetriever` (alpha-weighted dense+BM25) + optional cross-encoder re-ranker + metadata-filter builder. |
| `scripts/ingest.py` | CLI: fetch/sample → parse → (embed → upsert). Has an offline `--dry-run`. |
| `scripts/query.py` | CLI: hybrid query with facets (`--year-min`, `--journal`, `--mesh`, `--rerank`). |
| `tests/test_ingest.py` | Offline stdlib smoke tests (no network/keys/models). |
| `data/sample/pubmed_sample.xml` | **Synthetic** PubMed-XML fixture (labelled; not real citations). |

## Key design decisions

- **Framework-free ingestion core.** `ingest.py` uses only the standard library
  (`urllib`, `xml.etree`). The heavy frameworks live in the layers above it, so
  the parser and document builder are unit-testable with no network, no API keys,
  and no model downloads. That's why `pytest` runs in ~20 ms anywhere.
- **Section-aware chunking.** A structured abstract's BACKGROUND / METHODS /
  RESULTS / CONCLUSIONS blocks each become their own node (carrying a `section`
  metadata field). A query about *results* isn't diluted by *methods* text
  sharing a chunk. Full-text bodies use a hierarchical parser for auto-merging.
- **Hybrid search over `dotproduct`.** Dense (semantic) + BM25 (lexical, great for
  gene/drug tokens like *SGLT2*, *STAT3*) vectors are blended by `alpha` at query
  time. BM25 is `fit()` on the corpus and its params saved next to the index so
  queries use the same IDF weights.
- **Namespaces per doc type.** Abstracts and full-text bodies live in separate
  namespaces inside one index, so retrieval can target either.
- **Fine-tuning seams.** `EMBED_MODEL` (config) → swap in your fine-tuned
  bi-encoder checkpoint with no index-code change. The re-ranker in `retrieve.py`
  is the hook where recall→precision lift gets measured.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # then fill in NCBI + Pinecone keys
```

An NCBI API key (free) raises the rate limit from 3→10 req/s. Pinecone's free
serverless starter tier is enough to run this.

## Usage

**Offline dry run** — parses the bundled synthetic fixture and builds nodes.
Needs only `llama-index-core` (no keys, no Pinecone, no model download):

```bash
python scripts/ingest.py --source sample --dry-run
```

**Live ingest** — fetch from PubMed and index into Pinecone:

```bash
python scripts/ingest.py --source ncbi --query "glymphatic system amyloid clearance" --retmax 200
```

**Query** — hybrid retrieval with facet filters and optional re-ranking:

```bash
python scripts/query.py "does IL-6 drive macrophage polarization?" --top-k 5
python scripts/query.py "SGLT2 heart failure outcomes" --alpha 0.3 --year-min 2020 --rerank
```

`--alpha 1.0` is pure dense, `--alpha 0.0` is pure BM25; 0.3–0.6 is a good
starting sweep.

## Tests

```bash
pytest tests/ -q     # 8 offline stdlib tests
```

## What's verified vs. what needs services

- **Runs offline today:** config, ingestion (XML parsing, document building),
  LlamaIndex node parsing, the `--dry-run` CLI, and the full test suite.
- **Needs API keys + ML deps (torch, sentence-transformers, pinecone):** dense
  embedding, BM25 upsert, Pinecone index creation, and live hybrid query. The
  code is written and compile-checked; supply keys and `pip install -r
  requirements.txt` to run it live.

## Roadmap (next components)

1. **Retrieval eval harness** — recall@k / MRR on a labelled query–passage set;
   this is the number the fine-tuned bi-encoder and re-ranker are measured against.
2. **Bi-encoder fine-tuning** — contrastive/triplet training on domain
   query–passage pairs; swap the checkpoint into `EMBED_MODEL`.
3. **Cross-encoder re-ranker** — already wired in `retrieve.py`; quantify the
   recall→precision lift.
4. **LoRA/QLoRA generator** — domain answer synthesis + citation formatting over
   retrieved nodes.
5. **RAGAS harness** — faithfulness, answer relevancy, context precision/recall,
   plus a small human-labelled set.

## Note on the sample data

`data/sample/pubmed_sample.xml` is **synthetic** — fabricated records in a
clearly-fake PMID range (`9000000x`) for pipeline testing only. They are not real
citations and must not be treated as real science. Replace with live `efetch`
output for real use.
