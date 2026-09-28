# Architecture

## Data flow

1. Source `.docx` files are stored in `data/legal_docs/`.
2. `chunking.py` extracts text and parses legal hierarchy.
3. Each chunk receives document/article/clause/point metadata and retrieval aliases.
4. `embedding.py` encodes `embed_text` with BGE-M3.
5. Embeddings are stored in a local Qdrant collection.
6. `rag.py` performs dense retrieval + BM25.
7. RRF combines both rankings.
8. BGE reranker scores the fused candidates.
9. Parent-context expansion reconstructs the relevant legal clause.
10. The LLM receives only the selected legal context and a citation-oriented prompt.

## Design decisions

### Why hybrid retrieval?

Legal questions often contain exact terms such as article numbers, vehicle categories, penalty phrases, and legal terminology. Dense retrieval captures semantic similarity, while BM25 preserves exact lexical matches.

### Why parent-context expansion?

A single point-level chunk can be insufficient to answer a legal question. After ranking, the system reconstructs related chunks belonging to the same legal parent so the LLM sees enough context without making the retrieval stage unnecessarily broad.

### Why keep the Qdrant index out of Git?

The index is a generated binary artifact and can be rebuilt deterministically from the versioned semantic chunks. Keeping it out of source control makes the repository smaller and avoids machine-specific state.

## Production gaps

For production deployment, the next engineering steps should include:

- document version/effective-date management
- official source URL and provenance fields
- citation validation before generation
- automated regression tests for known legal questions
- retrieval and answer evaluation on temporally separated data
- persistent/managed Qdrant instead of a local embedded database
- authentication, rate limiting, observability, and request tracing
