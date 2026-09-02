# biomed-rag

A domain-expert RAG pipeline over **biomedical literature** (PubMed / PMC).

- **Component 1** — ingestion + hybrid indexing + retrieval (LlamaIndex + Pinecone).
- **Component 2** — a retriever-agnostic **evaluation harness** (IR metrics + a
  labelled benchmark with an easy / hard-paraphrase split) and a **grounded
  answer generator** with inline PMID citations and reference-free grounding
  metrics.

Both run **fully offline** against a synthetic corpus — no API keys, no model
download — so the whole thing is testable in CI. The remaining components (a
fine-tuned bi-encoder, a measured cross-encoder re-ranker, a LoRA-tuned
generator) drop into seams that already exist here, and Component 2 is the ruler
they're measured with.

> Resume framing: *"Built a production RAG pipeline (LlamaIndex + Pinecone) with
> hybrid dense+BM25 retrieval and section-aware biomedical chunking, plus a
> retriever-agnostic evaluation harness (Recall/Precision/F1/Hit@k, MRR, MAP,
> nDCG, R-Precision) over a 56-query labelled benchmark with an adversarial
> paraphrase split, and a citation-grounded generator scored on faithfulness,
> answer relevancy and context precision/recall. The benchmark quantifies the
> lexical-baseline ceiling (Recall@10 1.00 in-vocabulary vs 0.50 on paraphrase)
> as the target the fine-tuned bi-encoder must close."*

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

## What's built (Component 2)

| Module | Responsibility |
|---|---|
| `biomed_rag/eval/metrics.py` | **Pure-stdlib** IR metrics: `precision/recall/f1/hit/ndcg @k`, `mrr`, `map`, `r_precision`. Graded relevance for nDCG. Hand-verifiable, no numpy. |
| `biomed_rag/eval/dataset.py` | Loads `qrels.jsonl` (labels keyed by **PMID**, not node id, so they survive chunking changes) + the benchmark corpus; validates every label resolves. |
| `biomed_rag/eval/offline.py` | `OfflineHybridRetriever` — Okapi BM25 + TF-IDF-cosine blended by `alpha`, same `retrieve()` contract as the Pinecone one. `LexicalReranker` stands in for the cross-encoder. Zero services. |
| `biomed_rag/eval/harness.py` | `evaluate(retrieve_fn, benchmark)` → `EvalReport` (aggregate + per-query + per-split); `compare()` renders alpha sweeps / ablations with Δ-vs-baseline. |
| `biomed_rag/generate/prompt.py` | Numbered, PMID-stamped context block + a system prompt that pins the model to context and tells it to abstain. |
| `biomed_rag/generate/llm.py` | `ExtractiveLLM` (deterministic select-and-cite, offline default) and `TransformersLLM` (local HF causal model, `local_files_only`). |
| `biomed_rag/generate/pipeline.py` | `AnswerPipeline`: retrieve → prompt → generate → extract citations; flags hallucinated citations and uncited sentences. |
| `biomed_rag/generate/evaluate.py` | Reference-free grounding proxies: faithfulness, answer relevancy, citation support, hallucinated-citation rate, context precision/recall. |
| `scripts/eval.py` | CLI: alpha sweep, split breakdown, re-ranker ablation, `--generation`, `--live`, JSON export. |
| `scripts/answer.py` | CLI: one grounded, cited answer for a question (`--model`, `--alpha`, `--rerank`, `--live`). |
| `data/benchmarks/corpus.xml` | **Synthetic** 24-doc corpus in overlapping topic clusters (near-duplicate distractors). |
| `data/benchmarks/qrels.jsonl` | 56 labelled queries: 42 `easy` (in-vocabulary) + 14 `hard` (pure paraphrase, near-zero lexical overlap). |

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
- **Retriever-agnostic eval.** The harness scores anything with a
  `retrieve(query, top_k) -> [RetrievalResult]` signature, so the offline
  lexical baseline, the live Pinecone retriever, and a future fine-tuned
  bi-encoder are all measured by the exact same code path.
- **Labels keyed by PMID.** Relevance judgements point at documents, not node
  ids — change the chunker and the benchmark still holds. The harness maps each
  retrieved node back to its PMID and de-dupes to a document ranking before
  scoring.
- **Easy / hard split.** The hard split is 14 pure-paraphrase queries with
  near-zero lexical overlap with their target abstract. Lexical retrieval is
  *supposed* to miss them — that split is the regression gate for the dense
  retriever, and the number a fine-tune has to move.
- **Grounding without an LLM judge.** `generate/evaluate.py` approximates the
  RAGAS metrics with token-overlap heuristics so they run offline and
  deterministically. Directional, not the published RAGAS numbers.

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

**Evaluate retrieval** — offline alpha sweep, easy/hard split breakdown, and
re-ranker ablation over the labelled benchmark (no keys, no model download):

```bash
python scripts/eval.py --rerank
python scripts/eval.py --alphas 0,0.25,0.5,0.75,1.0 --k 1,3,5,10 --out data/benchmarks/results.json
python scripts/eval.py --generation                 # + grounded-answer metrics
python scripts/eval.py --live --artifacts-dir data/cache --rerank   # score the real Pinecone retriever
```

**Grounded answer** — retrieve + generate one cited answer:

```bash
python scripts/answer.py "does IL-6 drive M2 macrophage polarization via STAT3?"
python scripts/answer.py "how is amyloid cleared from the brain during sleep?" --alpha 0.3 --top-k 4
python scripts/answer.py "semaglutide vs tirzepatide for weight loss" --model HuggingFaceTB/SmolLM2-135M-Instruct
```

## Results (offline lexical baseline)

`python scripts/eval.py --rerank --generation` over the 56-query benchmark /
24-doc synthetic corpus. These are the baseline numbers a fine-tuned bi-encoder
and a real cross-encoder are meant to beat:

| Split | Recall@5 | Recall@10 | nDCG@10 | MRR |
|---|---|---|---|---|
| easy (n=42, in-vocabulary) | 1.00 | 1.00 | 0.98 | 0.98 |
| hard (n=14, pure paraphrase) | 0.50 | 0.50 | 0.38 | 0.34 |

The alpha sweep is nearly flat offline — both channels are lexical, so BM25 and
TF-IDF largely agree; `alpha` separation is what the real dense retriever adds.

Grounded generation (offline `ExtractiveLLM`): faithfulness **1.00**, citation
support **1.00**, hallucinated-citation rate **0.00**, context-recall **0.83**,
context-precision **0.41**, abstention **3.6%**.

## Tests

```bash
pytest tests/ -q     # 32 offline tests (~0.5s): ingest, IR metrics, benchmark
                     # integrity, eval harness, grounded-generation pipeline
```

## What's verified vs. what needs services

- **Runs offline today:** config, ingestion, LlamaIndex node parsing, the
  `--dry-run` CLI, **the full retrieval eval harness** (`scripts/eval.py`),
  **grounded generation with the `ExtractiveLLM`** (`scripts/answer.py`), and the
  32-test suite.
- **Needs a cached model (no keys):** `TransformersLLM` abstractive generation
  (`--model <hf-id>`). Wired and exercised; a 135M instruct model is too weak to
  cite well — the seam is for a LoRA-tuned biomedical generator.
- **Needs API keys + `pip install -r requirements.txt`:** dense embedding, BM25
  upsert, Pinecone index creation, live hybrid query, `scripts/eval.py --live`,
  and the real `CrossEncoderReranker`.

## Roadmap

1. ✅ **Retrieval eval harness** — `biomed_rag/eval` (metrics, benchmark, harness).
2. **Bi-encoder fine-tuning** — contrastive/triplet training on domain
   query–passage pairs; swap the checkpoint into `EMBED_MODEL`, re-run
   `scripts/eval.py` and watch the **hard split** move.
3. **Cross-encoder re-ranker** — wired in `retrieve.py`; `scripts/eval.py
   --rerank --live` quantifies the recall→precision lift.
4. ◑ **LoRA/QLoRA generator** — pipeline, prompt, citation extraction and
   grounding metrics are done (`biomed_rag/generate`); the tuned generator drops
   in via `--model` / `GEN_MODEL`.
5. **RAGAS harness** — replace the heuristic grounding proxies in
   `generate/evaluate.py` with the real RAGAS metrics + a human-labelled set.

## Note on the synthetic data

`data/sample/pubmed_sample.xml` (`9000000x` PMIDs) and
`data/benchmarks/corpus.xml` (`910000xx` PMIDs) are **synthetic** — fabricated
records for offline pipeline testing only. They are not real citations and must
not be treated as real science. Replace with live `efetch` output for real use.
