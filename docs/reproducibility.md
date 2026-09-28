# Reproducibility

## Environment

Use Python 3.10+.

```bash
python -m venv .venv
```

Install core dependencies:

```bash
pip install -r requirements.txt
```

For RAGAS evaluation:

```bash
pip install -r requirements-eval.txt
```

## Step 1 — Build chunks

```bash
python -m scripts.build_chunks
```

The parser reads `data/legal_docs/` and writes:

```text
data/processed/semantic_chunks.json
```

The repository includes the current generated file so a fresh checkout does not need to rebuild it unless the source documents change.

## Step 2 — Build the vector index

```bash
python -m scripts.build_index
```

The first run downloads the embedding model.

The generated Qdrant index is written to:

```text
data/qdrant/
```

It is ignored by Git.

## Step 3 — Start an LLM

The default setup assumes Ollama or another OpenAI-compatible endpoint:

```text
http://localhost:11434/v1
```

Configure the endpoint through `.env`.

## Step 4 — Start the API

```bash
python -m scripts.run_api
```

Test:

```bash
curl http://localhost:8000/health
```

## Step 5 — Start the UI

```bash
python -m scripts.run_web
```

Open `http://localhost:7860`.

## Step 6 — Run retrieval benchmark

Example:

```bash
python -m scripts.run_benchmark --limit 400 --top-k 10
```

Add `--with-llm` to evaluate answer generation as well.

## Step 7 — Run ablation

RAG only:

```bash
python -m scripts.run_ablation_eval --rag-only --limit 400
```

No-RAG only:

```bash
python -m scripts.run_ablation_eval --no-rag-only --limit 400
```

## Evaluation notes

The benchmark artifacts currently committed to the repository are historical snapshots from the team project. They are useful for portfolio/research documentation, but they should not be interpreted as a guarantee of current performance.

When changing:

- source documents
- chunking rules
- embedding model
- retrieval parameters
- reranker
- LLM
- prompt

rerun the benchmark and update the evaluation snapshot.
