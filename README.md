# Vietnamese Traffic Law RAG

A grounded question-answering system for **Vietnamese traffic law**, built as a reproducible NLP/RAG project.

The system combines:

- hierarchical legal-document chunking
- **BGE-M3** dense embeddings
- local **Qdrant** vector search
- lexical **BM25** retrieval
- **Reciprocal Rank Fusion (RRF)**
- **BGE reranking**
- parent-context expansion for legal clauses
- an OpenAI-compatible LLM endpoint (tested with local Ollama)
- FastAPI and Gradio interfaces
- retrieval and generation evaluation

> **Important:** This project is an academic/research prototype, not a substitute for professional legal advice. Legal documents and evaluation results are snapshots and should be revalidated before real-world use.

## Why this project?

Legal QA is not just a semantic-search problem. A useful system needs to preserve the legal hierarchy — **document → chapter → article → clause → point** — while retrieving enough context to support a grounded answer.

This project therefore evaluates a hybrid retrieval pipeline rather than relying on a single vector search method.

## Architecture

```text
                         ┌──────────────────────┐
                         │ Vietnamese legal docs│
                         │       (.docx)         │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │ Hierarchical chunking│
                         │ Article / Clause /   │
                         │ Point + metadata     │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │     BGE-M3           │
                         │      embeddings      │
                         └──────────┬───────────┘
                                    │
                                    ▼
                         ┌──────────────────────┐
                         │       Qdrant         │
                         │   dense retrieval    │
                         └──────────┬───────────┘
                                    │
                   ┌────────────────┴────────────────┐
                   │                                 │
                   ▼                                 ▼
             Dense retrieval                    BM25 retrieval
                   │                                 │
                   └────────────────┬────────────────┘
                                    ▼
                              RRF fusion
                                    │
                                    ▼
                              BGE reranker
                                    │
                                    ▼
                           Parent-context expansion
                                    │
                                    ▼
                              Grounded LLM
                                    │
                         ┌──────────┴──────────┐
                         ▼                     ▼
                     FastAPI                Gradio
```

## Repository structure

```text
.
├── data/
│   ├── legal_docs/          # Source legal documents used by the demo
│   ├── processed/           # Semantic chunks used by the pipeline
│   └── benchmark/           # Versioned benchmark datasets
├── evaluation/
│   └── results/             # Compact evaluation summaries
├── scripts/
│   ├── build_chunks.py      # Parse legal documents into semantic chunks
│   ├── build_index.py       # Build the local Qdrant index
│   ├── run_benchmark.py    # Retrieval benchmark
│   ├── run_ablation_eval.py# RAG vs. no-RAG evaluation
│   ├── run_api.py           # FastAPI server
│   └── run_web.py           # Gradio UI
├── src/
│   └── traffic_law_rag/
│       ├── api.py
│       ├── chunking.py
│       ├── config.py
│       ├── embedding.py
│       ├── qdrant.py
│       ├── rag.py
│       ├── reranking.py
│       ├── tokenizer.py
│       └── web.py
├── tests/
├── docs/
├── .env.example
├── pyproject.toml
└── README.md
```

## Quick start

### 1. Create an environment

```bash
python -m venv .venv
```

Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### 2. Configure the LLM

Copy the environment template:

```bash
copy .env.example .env
```

or on Linux/macOS:

```bash
cp .env.example .env
```

The default configuration expects an OpenAI-compatible local endpoint:

```text
LLM_BASE_URL=http://localhost:11434/v1
LLM_API_KEY=ollama
LLM_MODEL=qwen2.5:7b
```

For Ollama, make sure the selected model is available locally.

### 3. Build semantic chunks

The repository already includes a versioned `data/processed/semantic_chunks.json`, so this step is only needed when the source documents change.

```bash
python -m scripts.build_chunks
```

The output is:

```text
data/processed/semantic_chunks.json
```

### 4. Build the Qdrant index

```bash
python -m scripts.build_index
```

This downloads `BAAI/bge-m3` on first use and creates a persistent local Qdrant database under:

```text
data/qdrant/
```

The database is intentionally ignored by Git because it is a generated binary artifact.

### 5. Run the API

```bash
python -m scripts.run_api
```

Then open:

```text
http://localhost:8000/docs
```

### 6. Run the web UI

```bash
python -m scripts.run_web
```

Open:

```text
http://localhost:7860
```

## Retrieval pipeline

For each user question:

1. Normalize Vietnamese text and expand selected legal-domain queries.
2. Retrieve candidates with dense BGE-M3 search.
3. Retrieve candidates with BM25.
4. Fuse both ranked lists using RRF.
5. Apply domain-specific candidate filtering where appropriate.
6. Rerank candidates with `BAAI/bge-reranker-v2-m3`.
7. Expand the selected chunks to their parent legal context.
8. Construct a citation-aware context.
9. Ask the LLM to answer **only from the retrieved legal context**.

The prompt explicitly instructs the model not to invent legal articles, clauses, points, or penalty amounts.

## Evaluation snapshot

The current source archive contains an existing benchmark snapshot with **1,452 indexed chunks and 400 QA samples**.

| Retrieval method | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
|---|---:|---:|---:|---:|
| BM25 only | 16.0% | 29.0% | 33.75% | 0.245 |
| Vector only | 44.25% | 62.5% | 67.25% | 0.549 |
| **Hybrid RRF + rerank** | **53.5%** | **72.75%** | **79.5%** | **0.644** |

The archived RAGAS snapshot reports:

- Faithfulness: **0.355**
- Answer correctness: **0.467**

These numbers are **evaluation snapshots, not universal performance claims**. They should be rerun after changing the corpus, retrieval parameters, reranker, LLM, or evaluation dataset.

See [`evaluation/results/evaluation_summary.json`](evaluation/results/evaluation_summary.json).

## Configuration

All runtime configuration is centralized in `src/traffic_law_rag/config.py` and can be overridden through `.env`.

Important variables:

```text
CHUNKS_PATH
QDRANT_PATH
QDRANT_COLLECTION
EMBEDDING_MODEL
RERANKER_MODEL
LLM_BASE_URL
LLM_API_KEY
LLM_MODEL
TOP_K
RETRIEVAL_K
RERANK_CANDIDATES
RRF_K
```

No machine-specific Windows paths are required.

## Reproducibility

Detailed setup and benchmark instructions are available in:

- [`docs/reproducibility.md`](docs/reproducibility.md)
- [`docs/architecture.md`](docs/architecture.md)

Run unit tests with:

```bash
pytest
```

## Research / engineering highlights

This project demonstrates:

- structured parsing of Vietnamese legal documents
- metadata-aware retrieval
- dense + lexical hybrid retrieval
- rank fusion
- neural reranking
- parent-context reconstruction
- grounded generation
- retrieval evaluation with Hit@K and MRR
- answer evaluation with RAGAS
- API and web deployment

## Limitations

- The included legal corpus is a project snapshot, not a continuously updated legal database.
- The local Qdrant index is generated and therefore not committed.
- LLM quality depends on the selected model and prompt.
- RAGAS results are model- and dataset-dependent.
- Retrieval metrics measure benchmark alignment, not legal correctness in every real-world case.
- Before production use, the system needs document versioning, effective-date handling, source provenance, stronger citation validation, and a more comprehensive evaluation protocol.

## License

This repository is intended for academic and portfolio use. See [`LICENSE`](LICENSE).

## Contact

**Hoàng Nguyên Long**  
Email: `long.hoangnguyen0108@gmail.com`
