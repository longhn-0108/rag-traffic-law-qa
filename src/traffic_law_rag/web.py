"""Gradio web interface for the legal RAG service."""

from __future__ import annotations

import gradio as gr
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL, TOP_K
from .qdrant import get_qdrant_client
from .rag import retrieve, ask_llm, source_from_point

qdrant_client = get_qdrant_client()
embedding_model = SentenceTransformer(EMBEDDING_MODEL)


def chat_fn(message: str, history: list) -> str:
    message = (message or "").strip()
    if not message:
        return "Vui lòng nhập câu hỏi."

    try:
        points = retrieve(message, qdrant_client, embedding_model, limit=TOP_K)
        if not points:
            return "Không tìm thấy context phù hợp trong cơ sở dữ liệu."

        answer = ask_llm(message, points)
        sources = [source_from_point(i, point) for i, point in enumerate(points, 1)]

        source_text = "\n\n### Nguồn tham khảo\n" + "\n".join(
            f"- **[{s['index']}]** {s['citation']}" for s in sources
        )
        return answer + source_text
    except Exception as exc:
        return f"Lỗi khi xử lý câu hỏi: `{exc}`"


demo = gr.ChatInterface(
    fn=chat_fn,
    title="Vietnamese Traffic Law RAG",
    description="Hỏi đáp grounded trên các văn bản pháp luật giao thông được nạp vào hệ thống.",
    textbox=gr.Textbox(
        placeholder="Ví dụ: Xe 50cc vượt đèn đỏ thì bị xử lý thế nào?",
        container=False,
    ),
)


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
