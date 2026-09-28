"""FastAPI application for the legal RAG service."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL, COLLECTION_NAME, TOP_K
from .qdrant import get_qdrant_client
from .rag import answer_question_text

app = FastAPI(
    title="Vietnamese Traffic Law RAG API",
    version="1.0.0",
    description="Grounded Q&A over Vietnamese traffic-law documents.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


qdrant_client = get_qdrant_client()
embedding_model = SentenceTransformer(EMBEDDING_MODEL)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "collection": COLLECTION_NAME}


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    question = request.question.strip()
    if not question:
        return {"answer": "Câu hỏi không được để trống.", "sources": []}
    return answer_question_text(question, qdrant_client, embedding_model)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("traffic_law_rag.api:app", host="0.0.0.0", port=8000, reload=False)
