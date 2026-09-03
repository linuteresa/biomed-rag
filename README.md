# biomed-rag

A production-shaped **RAG pipeline over biomedical literature** (PubMed / PMC):
ingest → section-aware chunk → hybrid (dense + BM25) index → retrieve → evaluate →
answer with citations.

Everything except live PubMed/Pinecone calls **runs offline against a synthetic
corpus** — no API keys, no model download — so the pipeline, the eval harness,
and the 32-test suite all run in CI in under a second.

```
PubMed E-utilities ─┐
                    ├─► BiomedDocument ─► LlamaIndex nodes ─► Pinecone hybrid index ─┐
data/sample/*.xml ──┘   (stdlib parse)   (section-aware)     (dense + BM25, dotproduct)│
                                                                                      ▼
              cited answer  ◄─── generate ◄─── HybridRetriever (alpha-weighted, +rerank)
                    │                                     │
              grounding metrics                     eval harness ──► recall / ranking-quality / first-hit-score
              (faithfulness, …)                     (labelled benchmark, easy/hard split)
```

---

## Components

### 1 · Ingestion + hybrid retrieval

| Module | Responsibility |
|---|---|
| `biomed_rag/config.py` | `.env`-driven config (NCBI, Pinecone, embed model). No import-time side effects. |
| `biomed_rag/ingest.py` | **Stdlib-only** NCBI E-utilities client (esearch/efetch, rate-limited, retrying) + PubMed-XML parser + `BiomedDocument`. |
| `biomed_rag/parse.py` | LlamaIndex layer. Structured-abstract sections (BACKGROUND/METHODS/RESULTS/…) each become their own node; full text uses `HierarchicalNodeParser`. |
| `biomed_rag/embeddings.py` | Pluggable dense bi-encoder + BM25 sparse encoder + `alpha` hybrid scaling. |
| `biomed_rag/index.py` | Pinecone hybrid store: serverless `dotproduct` index, per-doc-type namespaces, sparse-dense upsert, metadata filters. |
| `biomed_rag/retrieve.py` | `HybridRetriever` (alpha-weighted dense+BM25) + optional `CrossEncoderReranker` + facet-filter builder. |

### 2 · Evaluation + grounded generation

| Module | Responsibility |
|---|---|
| `biomed_rag/eval/metrics.py` | Pure-stdlib retrieval metrics: precision, recall, f-score, any-hit, ranking-quality (nDCG), first-hit-score (MRR), avg-precision (MAP), precision-at-N — graded relevance, hand-verifiable. `display_name()` / `LEGEND` give the readable labels used in output. |
| `biomed_rag/eval/dataset.py` | Loads `qrels.jsonl` (labels keyed by **PMID**, not node id) + benchmark corpus; validates every label resolves. |
| `biomed_rag/eval/offline.py` | `OfflineHybridRetriever` — Okapi BM25 + TF-IDF-cosine blended by `alpha`, **same `retrieve()` contract as Pinecone**. `LexicalReranker` stands in for the cross-encoder. |
| `biomed_rag/eval/harness.py` | `evaluate(retrieve_fn, benchmark)` → `EvalReport` (aggregate + per-query + per-split); `compare()` renders sweeps/ablations with Δ-vs-baseline. |
| `biomed_rag/generate/prompt.py` | Numbered, PMID-stamped context block + a system prompt that pins the model to context and tells it to abstain. |
| `biomed_rag/generate/llm.py` | `ExtractiveLLM` (deterministic select-and-cite, offline default) · `TransformersLLM` (local HF causal model, `local_files_only`). |
| `biomed_rag/generate/pipeline.py` | `AnswerPipeline`: retrieve → prompt → generate → extract citations; flags hallucinated citations + uncited sentences. |
| `biomed_rag/generate/evaluate.py` | Reference-free grounding proxies: faithfulness, answer relevancy, citation support, hallucinated-citation rate, context precision/recall. |

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest -q                                   # 32 offline tests, ~0.5s
```

### Offline (no keys, no downloads)

```bash
# parse the synthetic fixture and build nodes
python scripts/ingest.py --source sample --dry-run

# retrieval eval: alpha sweep + easy/hard split + re-ranker ablation
python scripts/eval.py --rerank --generation

# one grounded, cited answer
python scripts/answer.py "does IL-6 drive M2 macrophage polarization via STAT3?"
```

### Live (needs `.env` with NCBI + Pinecone keys)

```bash
cp .env.example .env      # fill in NCBI_API_KEY, PINECONE_API_KEY, …

python scripts/ingest.py --source ncbi --query "glymphatic amyloid clearance" --retmax 200
python scripts/query.py  "SGLT2 heart failure outcomes" --alpha 0.3 --year-min 2020 --rerank
python scripts/eval.py   --live --artifacts-dir data/cache --rerank
python scripts/answer.py "carbapenem-resistant Klebsiella treatment" --live --rerank
```

`--alpha 1.0` = pure dense, `--alpha 0.0` = pure BM25; 0.3–0.6 is a good sweep.
Point `--model <hf-id>` / `GEN_MODEL` at a cached instruct checkpoint for
abstractive (vs extractive) answers.

---

## Benchmark & baseline results

`data/benchmarks/` — a **synthetic 24-doc corpus** in overlapping topic clusters
(near-duplicate distractors) and **56 labelled queries**:

- **easy** (42) — natural queries that share vocabulary with the target; includes
  acronym/gene-token cases and within-cluster disambiguation.
- **hard** (14) — pure paraphrase, near-zero lexical overlap with the target
  abstract. Lexical retrieval is *meant* to miss these — this split is the
  regression gate for a fine-tuned dense retriever.

`python scripts/eval.py --rerank --generation` (offline BM25 + TF-IDF baseline):

| Split | recall@5 | recall@10 | ranking-quality@10 | first-hit-score |
|---|---|---|---|---|
| easy (in-vocabulary) | 1.00 | 1.00 | 0.98 | 0.98 |
| hard (paraphrase) | 0.50 | 0.50 | 0.38 | 0.34 |

`ranking-quality` is nDCG (0–1, how close the order is to ideal);
`first-hit-score` is mean reciprocal rank (1 / rank of the first correct hit).
`scripts/eval.py` prints a one-line key for every column.

Grounded generation (offline `ExtractiveLLM`): faithfulness **1.00**, citation
support **1.00**, hallucinated-citation rate **0.00**, context-recall **0.83**,
context-precision **0.41**, abstention **3.6%**.

> The alpha sweep is ~flat offline (BM25 and TF-IDF are both lexical and agree);
> `alpha` separation is what a real dense retriever adds.

---

## Design decisions

- **Framework-free ingestion core.** `ingest.py` is pure `urllib` + `xml.etree`;
  the heavy frameworks live in the layers above, so parsing/document-building is
  unit-testable with no network, keys, or model downloads.
- **Section-aware chunking.** A query about *results* isn't diluted by *methods*
  text sharing a chunk; each node carries a `section` metadata field.
- **Hybrid over `dotproduct`.** Dense (semantic) + BM25 (lexical, strong on
  gene/drug tokens like *SGLT2*, *STAT3*) blended by `alpha` at query time; BM25
  `fit()` on the corpus and its IDF params saved next to the index.
- **Retriever-agnostic eval.** The harness scores anything with a
  `retrieve(query, top_k) -> [RetrievalResult]` signature — offline baseline,
  live Pinecone, and a future fine-tuned bi-encoder all go through one code path.
- **Labels keyed by PMID.** Relevance judgements point at documents, not node ids;
  change the chunker and the benchmark still holds.
- **Grounding without an LLM judge.** `generate/evaluate.py` approximates the
  RAGAS metrics with token-overlap heuristics so they run offline and
  deterministically — directional, not the published RAGAS numbers.
- **Fine-tuning seams.** `EMBED_MODEL` swaps the bi-encoder with no index-code
  change; `--model` / `GEN_MODEL` swaps the generator; `--rerank` is where the
  recall→precision lift gets measured.

---

## Project layout

```
biomed_rag/
  config.py  ingest.py  parse.py  embeddings.py  index.py  retrieve.py
  eval/      metrics.py  dataset.py  offline.py  harness.py
  generate/  prompt.py   llm.py      pipeline.py  evaluate.py
scripts/     ingest.py  query.py  eval.py  answer.py
data/
  sample/pubmed_sample.xml          # synthetic fixture for Component 1 tests
  benchmarks/corpus.xml, qrels.jsonl # synthetic labelled eval set
tests/       test_ingest.py  test_metrics.py  test_benchmark_data.py
             test_eval_harness.py  test_generation.py
```

---

## Status

| | Runs offline today | Needs a cached model | Needs API keys + full deps |
|---|---|---|---|
| **Works** | config, ingestion, node parsing, `--dry-run`, full eval harness, extractive grounded answers, 32 tests | `TransformersLLM` abstractive generation (`--model`) | dense embedding, BM25 upsert, Pinecone index, live query, `eval.py --live`, real `CrossEncoderReranker` |

### Roadmap

1. ✅ **Retrieval eval harness** — `biomed_rag/eval`.
2. **Bi-encoder fine-tuning** — contrastive/triplet on domain query–passage pairs;
   swap into `EMBED_MODEL`, re-run `eval.py`, watch the **hard split** move.
3. **Cross-encoder re-ranker** — wired; `eval.py --rerank --live` quantifies the lift.
4. ◑ **LoRA/QLoRA generator** — pipeline, prompt, citation extraction and grounding
   metrics are done; the tuned model drops in via `--model` / `GEN_MODEL`.
5. **RAGAS harness** — replace the heuristic grounding proxies with real RAGAS
   metrics + a human-labelled set.

---

## Note on the synthetic data

`data/sample/pubmed_sample.xml` (`9000000x` PMIDs) and `data/benchmarks/corpus.xml`
(`910000xx` PMIDs) are **fabricated** for offline testing only. They are not real
citations and must not be treated as real science. Replace with live `efetch`
output for real use.
