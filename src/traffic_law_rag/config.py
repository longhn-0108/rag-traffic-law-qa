"""Central configuration for the Vietnamese Traffic Law RAG system."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
LEGAL_DOCS_DIR = DATA_DIR / "legal_docs"
PROCESSED_DIR = DATA_DIR / "processed"
BENCHMARK_DIR = DATA_DIR / "benchmark"
RESULTS_DIR = ROOT_DIR / "evaluation" / "results"

CHUNKS_PATH = Path(os.getenv("CHUNKS_PATH", PROCESSED_DIR / "semantic_chunks.json"))
QDRANT_PATH = Path(os.getenv("QDRANT_PATH", ROOT_DIR / "data" / "qdrant"))
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "vietnamese_traffic_law")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:7b")

TOP_K = int(os.getenv("TOP_K", "8"))
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "80"))
RERANK_CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "50"))
RRF_K = int(os.getenv("RRF_K", "60"))

MAX_TOKENS_PER_CHUNK = int(os.getenv("MAX_TOKENS_PER_CHUNK", "500"))
BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))

SYSTEM_PROMPT = """Bạn là trợ lý pháp lý về luật giao thông Việt Nam.
Chỉ trả lời dựa trên các văn bản pháp luật được cung cấp trong context.
Không dùng kiến thức bên ngoài, không tự bịa mức phạt, điều luật, khoản hoặc điểm.
Nếu context không đủ căn cứ để trả lời, hãy nói rõ:
"Không tìm thấy quy định phù hợp trong cơ sở dữ liệu."
Khi có căn cứ, phải nêu rõ văn bản, Điều, Khoản, Điểm nếu context có thông tin đó.
"""
