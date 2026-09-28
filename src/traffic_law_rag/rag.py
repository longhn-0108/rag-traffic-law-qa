from __future__ import annotations

import json
import math
import re
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from openai import OpenAI
    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

from .config import (
    CHUNKS_PATH,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LLM_API_KEY,
    LLM_BASE_URL,
    LLM_MODEL,
    RETRIEVAL_K,
    RERANK_CANDIDATES,
    RRF_K,
    SYSTEM_PROMPT,
    TOP_K,
)
from .reranking import rerank_points

MODEL_NAME = EMBEDDING_MODEL
EXIT_COMMANDS = {"exit", "quit", "q", "thoat", "thoát"}
TOKEN_RE = re.compile(r"[0-9a-zA-ZÀ-ỹ/.-]+")

SYSTEM_PROMPT = """Bạn là trợ lý pháp lý về luật giao thông Việt Nam.
Chỉ trả lời dựa trên các văn bản pháp luật được cung cấp trong context.
Không dùng kiến thức bên ngoài, không tự bịa mức phạt, điều luật, khoản hoặc điểm.
Nếu context không đủ căn cứ để trả lời, hãy nói rõ: "Không tìm thấy quy định phù hợp trong cơ sở dữ liệu."
Khi có căn cứ, phải nêu rõ văn bản, Điều, Khoản, Điểm nếu context có thông tin đó."""


@dataclass
class RetrievedPoint:
    payload: dict
    score: float = 0.0


def normalize_vietnamese(text: str) -> str:
    normalized = unicodedata.normalize("NFD", str(text))
    return "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    ).replace("đ", "d").replace("Đ", "D").lower()


def tokenize(text: str) -> list[str]:
    normalized = normalize_vietnamese(text)
    return [token for token in TOKEN_RE.findall(normalized) if token]


@lru_cache(maxsize=1)
def load_chunks() -> tuple[dict, ...]:
    path = Path(CHUNKS_PATH)
    with path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not isinstance(chunks, list):
        raise ValueError(f"{CHUNKS_PATH} phải là một JSON list.")

    return tuple(chunk for chunk in chunks if chunk.get("id") and chunk.get("embed_text"))


def payload_from_chunk(chunk: dict) -> dict:
    metadata = dict(chunk.get("metadata") or {})
    chunk_id = str(chunk["id"])
    content = str(chunk.get("content") or "")
    contextualized_content = str(chunk.get("contextualized_content") or content)
    embed_text = str(chunk.get("embed_text") or contextualized_content)

    metadata.setdefault("source", chunk_id)
    if metadata.get("heading_title") and not metadata.get("article_title"):
        metadata["article_title"] = metadata["heading_title"]

    return {
        "chunk_id": chunk_id,
        "source": chunk_id,
        "content": content,
        "contextualized_content": contextualized_content,
        "embed_text": embed_text,
        "page_content": contextualized_content,
        "metadata": metadata,
    }


@lru_cache(maxsize=1)
def chunk_payload_index() -> dict[str, dict]:
    return {str(chunk["id"]): payload_from_chunk(chunk) for chunk in load_chunks()}


@lru_cache(maxsize=1)
def parent_groups() -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for payload in chunk_payload_index().values():
        metadata = payload.get("metadata") or {}
        group_key = parent_group_key(payload)
        groups[group_key].append(payload)
    return dict(groups)


@lru_cache(maxsize=1)
def bm25_index() -> dict:
    docs = []
    inverted: dict[str, list[tuple[int, int]]] = defaultdict(list)
    doc_freq: Counter[str] = Counter()

    for idx, chunk in enumerate(load_chunks()):
        text = " ".join(
            str(part)
            for part in [
                chunk.get("embed_text", ""),
                " ".join((chunk.get("metadata") or {}).get("aliases") or []),
            ]
            if part
        )
        tokens = tokenize(text)
        counts = Counter(tokens)
        docs.append({"chunk": chunk, "length": len(tokens), "counts": counts})
        for term, tf in counts.items():
            doc_freq[term] += 1
            inverted[term].append((idx, tf))

    avgdl = sum(doc["length"] for doc in docs) / max(len(docs), 1)
    return {
        "docs": docs,
        "inverted": dict(inverted),
        "doc_freq": doc_freq,
        "avgdl": avgdl,
        "n_docs": len(docs),
    }


def bm25_search(query: str, limit: int) -> list[RetrievedPoint]:
    index = bm25_index()
    query_terms = Counter(tokenize(query))
    scores: defaultdict[int, float] = defaultdict(float)
    k1 = 1.5
    b = 0.75

    for term, query_tf in query_terms.items():
        postings = index["inverted"].get(term)
        if not postings:
            continue

        df = index["doc_freq"][term]
        idf = math.log(1 + (index["n_docs"] - df + 0.5) / (df + 0.5))
        for doc_idx, tf in postings:
            doc_len = index["docs"][doc_idx]["length"] or 1
            denom = tf + k1 * (1 - b + b * doc_len / index["avgdl"])
            scores[doc_idx] += query_tf * idf * (tf * (k1 + 1) / denom)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
    points = []
    for doc_idx, score in ranked:
        chunk = index["docs"][doc_idx]["chunk"]
        payload = payload_from_chunk(chunk)
        payload["_bm25_score"] = float(score)
        points.append(RetrievedPoint(payload=payload, score=float(score)))
    return points


def expand_query(question: str) -> str:
    normalized = normalize_vietnamese(question)
    expansions = []

    if any(term in normalized for term in ["den do", "vuot den", "chay den"]):
        expansions.extend([
            "không chấp hành hiệu lệnh của đèn tín hiệu giao thông",
            "đèn tín hiệu giao thông",
        ])

    if any(term in normalized for term in ["50cc", "50 cc", "xe 50", "xe may", "xe gan may", "mo to"]):
        expansions.extend([
            "xe mô tô",
            "xe gắn máy",
            "các loại xe tương tự xe mô tô và các loại xe tương tự xe gắn máy",
        ])

    return " ".join([question, *expansions]).strip()


def get_llm_client() -> OpenAI:
    from openai import OpenAI
    return OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)


def build_search_query(question: str) -> str:
    expanded_question = expand_query(question)
    search_query = "\n".join(
        part for part in [question, expanded_question] if part and part.strip()
    )
    return search_query


def dense_search(
    search_query: str,
    client: QdrantClient,
    model: SentenceTransformer,
    limit: int,
) -> list[RetrievedPoint]:
    query_vector = model.encode(search_query, normalize_embeddings=True).tolist()
    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )

    points = []
    for point in results.points:
        payload = dict(point.payload or {})
        payload["_vector_score"] = float(point.score or 0.0)
        points.append(RetrievedPoint(payload=payload, score=float(point.score or 0.0)))
    return points


def chunk_id_for_point(point: RetrievedPoint) -> str:
    payload = point.payload or {}
    metadata = payload.get("metadata") or {}
    return str(payload.get("chunk_id") or metadata.get("source") or payload.get("source") or "")


def rrf_fusion(
    dense_points: list[RetrievedPoint],
    bm25_points: list[RetrievedPoint],
    limit: int,
) -> list[RetrievedPoint]:
    payloads: dict[str, dict] = {}
    scores: defaultdict[str, float] = defaultdict(float)

    for rank, point in enumerate(dense_points, start=1):
        chunk_id = chunk_id_for_point(point)
        if not chunk_id:
            continue
        payloads.setdefault(chunk_id, dict(point.payload))
        payloads[chunk_id]["_vector_score"] = point.payload.get("_vector_score", point.score)
        scores[chunk_id] += 1.0 / (RRF_K + rank)

    for rank, point in enumerate(bm25_points, start=1):
        chunk_id = chunk_id_for_point(point)
        if not chunk_id:
            continue
        payloads.setdefault(chunk_id, dict(point.payload))
        payloads[chunk_id]["_bm25_score"] = point.payload.get("_bm25_score", point.score)
        scores[chunk_id] += 1.0 / (RRF_K + rank)

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
    fused = []
    for chunk_id, score in ranked:
        payload = payloads[chunk_id]
        payload["_rrf_score"] = float(score)
        fused.append(RetrievedPoint(payload=payload, score=float(score)))
    return fused


def filter_candidates_by_vehicle(query: str, points: list[RetrievedPoint], min_keep: int) -> list[RetrievedPoint]:
    normalized_query = normalize_vietnamese(query)
    asks_motorbike = any(
        term in normalized_query
        for term in ["xe may", "xe gan may", "mo to", "50cc", "50 cc"]
    )
    asks_specialized = "chuyen dung" in normalized_query

    if not asks_motorbike or asks_specialized:
        return points

    preferred = []
    for point in points:
        payload = point.payload or {}
        metadata = payload.get("metadata") or {}
        haystack = normalize_vietnamese(
            " ".join(
                str(part)
                for part in [
                    metadata.get("heading_title", ""),
                    payload.get("page_content", ""),
                    payload.get("content", ""),
                ]
            )
        )
        is_motorbike_group = "xe mo to" in haystack or "xe gan may" in haystack
        is_specialized_group = "xe may chuyen dung" in haystack
        if is_motorbike_group and not is_specialized_group:
            preferred.append(point)

    return preferred if len(preferred) >= min_keep else points


def parent_group_key(payload: dict) -> str:
    metadata = payload.get("metadata") or {}
    if metadata.get("parent_id"):
        return str(metadata["parent_id"])

    doc_id = metadata.get("document_id") or "unknown_doc"
    article = metadata.get("article")
    clause = metadata.get("clause")
    if article and clause:
        return f"{doc_id}_D{article}_K{clause}"
    if article:
        return f"{doc_id}_D{article}"
    return str(payload.get("chunk_id") or metadata.get("source") or "unknown")


def legal_reference(metadata: dict) -> str:
    document_type = metadata.get("document_type") or "Văn bản"
    document_id = metadata.get("document_id") or metadata.get("source") or "-"
    parts = [f"{document_type} {document_id}"]

    article = metadata.get("article")
    if article:
        article_text = f"Điều {article}"
        title = metadata.get("heading_title") or metadata.get("article_title")
        if title:
            article_text += f": {title}"
        parts.append(article_text)

    clause = metadata.get("clause")
    if clause:
        parts.append(f"Khoản {clause}")

    point = metadata.get("point")
    if point:
        parts.append(f"Điểm {point}")

    return " - ".join(parts)


def get_point_text(point: RetrievedPoint) -> str:
    payload = point.payload or {}
    return (
        payload.get("page_content")
        or payload.get("contextualized_content")
        or payload.get("content")
        or payload.get("embed_text")
        or ""
    )


def merge_group_payload(seed_payload: dict, group_payloads: Iterable[dict]) -> dict:
    group_list = list(group_payloads)
    seed_metadata = dict(seed_payload.get("metadata") or {})
    texts = []
    seen_texts = set()
    source_ids = []

    for payload in group_list:
        text = str(payload.get("content") or payload.get("page_content") or "").strip()
        if text and text not in seen_texts:
            texts.append(text)
            seen_texts.add(text)
        source_ids.append(str(payload.get("chunk_id") or (payload.get("metadata") or {}).get("source") or ""))

    if not texts:
        texts.append(str(seed_payload.get("content") or seed_payload.get("page_content") or ""))

    citation = legal_reference(seed_metadata)
    merged_content = "\n".join(texts)
    merged_payload = dict(seed_payload)
    merged_payload["page_content"] = f"{citation}\nNội dung:\n{merged_content}"
    merged_payload["content"] = merged_content
    merged_payload["_source_chunk_ids"] = [source_id for source_id in source_ids if source_id]
    merged_payload["_parent_group_key"] = parent_group_key(seed_payload)
    return merged_payload


def expand_parent_context(points: list[RetrievedPoint]) -> list[RetrievedPoint]:
    groups = parent_groups()
    expanded = []
    used_group_keys = set()

    for point in points:
        payload = point.payload or {}
        group_key = parent_group_key(payload)
        if group_key in used_group_keys:
            continue
        used_group_keys.add(group_key)

        group_payloads = groups.get(group_key) or [payload]
        merged_payload = merge_group_payload(payload, group_payloads)
        merged_payload["_vector_score"] = payload.get("_vector_score")
        merged_payload["_bm25_score"] = payload.get("_bm25_score")
        merged_payload["_rrf_score"] = payload.get("_rrf_score")
        merged_payload["_rerank_score_raw"] = payload.get("_rerank_score_raw")
        merged_payload["_rerank_score"] = payload.get("_rerank_score")
        expanded.append(RetrievedPoint(payload=merged_payload, score=point.score))

    return expanded


def build_context(points: list[RetrievedPoint]) -> str:
    blocks = []
    for i, point in enumerate(points, start=1):
        payload = point.payload or {}
        metadata = payload.get("metadata") or {}
        source_ids = ", ".join(payload.get("_source_chunk_ids") or [payload.get("chunk_id", "-")])
        blocks.append(
            f'<document index="{i}" source_chunks="{source_ids}">\n'
            f"<citation>{legal_reference(metadata)}</citation>\n"
            f"<content>\n{get_point_text(point)}\n</content>\n"
            f"</document>"
        )
    return "\n\n".join(blocks)


def build_prompt(question: str, points: list[RetrievedPoint]) -> str:
    context = build_context(points)
    return f"""<legal_context>
{context}
</legal_context>

<user_query>
{question}
</user_query>

Yêu cầu trả lời:
1. Chỉ dùng thông tin trong <legal_context>.
2. Nếu có quy định phù hợp, trả lời trực tiếp, rõ mức xử lý/mức phạt nếu context có nêu.
3. Bắt buộc trích căn cứ theo dạng: "Theo [Văn bản], Điều X, Khoản Y, Điểm Z...".
4. Nếu context không đủ thông tin, trả lời: "Không tìm thấy quy định phù hợp trong cơ sở dữ liệu."
5. Không nhắc đến tên kỹ thuật như RAG, chunk, vector, rerank."""


def retrieve(question: str, client: QdrantClient, model: SentenceTransformer, limit: int = TOP_K):
    search_query = build_search_query(question)

    print("\n=== SEARCH QUERY ===")
    print(f"Câu hỏi: {question}")
    if search_query != question:
        print(f"Query mở rộng: {search_query}")

    dense_points = dense_search(search_query, client, model, limit=RETRIEVAL_K)
    bm25_points = bm25_search(search_query, limit=RETRIEVAL_K)
    fused_points = rrf_fusion(dense_points, bm25_points, limit=RERANK_CANDIDATES)
    filtered_points = filter_candidates_by_vehicle(search_query, fused_points, min_keep=limit * 3)
    reranked_points = rerank_points(search_query, filtered_points, top_k=limit)
    return expand_parent_context(reranked_points)


def ask_llm(question: str, points) -> str:
    prompt = build_prompt(question, points)
    client = get_llm_client()

    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content
    return content.strip() if isinstance(content, str) else str(content)


def source_from_point(index: int, point: RetrievedPoint) -> dict:
    payload = point.payload or {}
    metadata = payload.get("metadata") or {}
    return {
        "index": index,
        "source": metadata.get("source", payload.get("chunk_id", "-")),
        "document": f"{metadata.get('document_type', 'Văn bản')} {metadata.get('document_id', '-')}",
        "article": metadata.get("article", "-"),
        "clause": metadata.get("clause", "-"),
        "point": metadata.get("point", "-"),
        "citation": legal_reference(metadata),
        "text": get_point_text(point)[:1000],
        "source_chunk_ids": payload.get("_source_chunk_ids", []),
        "vector_score": payload.get("_vector_score"),
        "bm25_score": payload.get("_bm25_score"),
        "rrf_score": payload.get("_rrf_score"),
        "rerank_score": payload.get("_rerank_score"),
    }


def answer_question_text(question: str, qdrant_client: QdrantClient, model: SentenceTransformer) -> dict:
    points = retrieve(question, qdrant_client, model, limit=TOP_K)
    if not points:
        return {"answer": "Không truy xuất được context nào.", "sources": []}

    try:
        answer = ask_llm(question, points)
    except Exception as exc:
        return {"answer": f"Không gọi được local LLM: {exc}", "sources": []}

    return {
        "answer": answer,
        "sources": [source_from_point(i, point) for i, point in enumerate(points, start=1)],
    }


def answer_question(question: str, qdrant_client: QdrantClient, model: SentenceTransformer) -> None:
    points = retrieve(question, qdrant_client, model, limit=TOP_K)
    if not points:
        print("Không truy xuất được context nào.")
        return

    print("\n=== TOP-K SAU HYBRID + RERANK + PARENT-CONTEXT ===")
    for i, point in enumerate(points, start=1):
        payload = point.payload or {}
        metadata = payload.get("metadata") or {}
        print(f"\n[{i}] {legal_reference(metadata)}")
        print(f"vector_score: {payload.get('_vector_score', '-')}")
        print(f"bm25_score: {payload.get('_bm25_score', '-')}")
        print(f"rrf_score: {payload.get('_rrf_score', '-')}")
        print(f"rerank_score: {payload.get('_rerank_score', '-')}")
        print(get_point_text(point)[:1500])

    print("\n=== LLM ANSWER ===")
    try:
        answer = ask_llm(question, points)
    except Exception as exc:
        print(f"Không gọi được local LLM: {exc}")
        return

    print(answer)


def main() -> None:
    qdrant_client = QdrantClient(path=QDRANT_PATH)
    model = SentenceTransformer(MODEL_NAME)

    try:
        if not qdrant_client.collection_exists(COLLECTION_NAME):
            print(f"Không tìm thấy collection '{COLLECTION_NAME}'")
            sys.exit(1)

        print("Nhập câu hỏi. Gõ 'exit', 'quit', 'q' hoặc 'thoát' để dừng.")
        while True:
            question = input("\nNhập câu hỏi của bạn: ").strip()
            if not question:
                print("Câu hỏi không được để trống.")
                continue

            if normalize_vietnamese(question) in EXIT_COMMANDS:
                print("Đã thoát.")
                break

            answer_question(question, qdrant_client, model)
    finally:
        qdrant_client.close()


if __name__ == "__main__":
    main()
