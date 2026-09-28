from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from traffic_law_rag.config import BENCHMARK_DIR, EMBEDDING_MODEL, LLM_MODEL
from traffic_law_rag.qdrant import get_qdrant_client
from traffic_law_rag.rag import ask_llm, get_llm_client, retrieve

MODEL_NAME = EMBEDDING_MODEL


DEFAULT_DATASET = str(BENCHMARK_DIR / "generated_rag_dataset.json")
DEFAULT_OUTPUT_DIR = "benchmark/results"
DEFAULT_RAG_OUTPUT = "rag_results.json"
DEFAULT_NO_RAG_OUTPUT = "no_rag_results.json"

NO_RAG_SYSTEM_PROMPT = """Bạn là trợ lý pháp lý về luật giao thông Việt Nam.
Trả lời trực tiếp câu hỏi của người dùng bằng kiến thức sẵn có của bạn.
Nếu không chắc chắn, hãy nói rõ là bạn không chắc thay vì bịa điều luật hoặc mức phạt."""


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def load_dataset(path: str) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Benchmark dataset phải là một JSON list.")

    return data


def source_ids_from_point(point) -> list[str]:
    payload = point.payload or {}
    metadata = payload.get("metadata") or {}
    source_ids = []

    for value in [
        payload.get("chunk_id"),
        payload.get("source"),
        metadata.get("source"),
    ]:
        if value:
            source_ids.append(str(value))

    for source_id in payload.get("_source_chunk_ids") or []:
        source_ids.append(str(source_id))

    seen = set()
    deduped = []
    for source_id in source_ids:
        if source_id not in seen:
            seen.add(source_id)
            deduped.append(source_id)
    return deduped


def source_ids_from_points(points) -> set[str]:
    source_ids = set()
    for point in points:
        source_ids.update(source_ids_from_point(point))
    return source_ids


def reciprocal_rank(points, expected_ids: set[str]) -> float:
    if not expected_ids:
        return 0.0

    for rank, point in enumerate(points, start=1):
        if expected_ids & set(source_ids_from_point(point)):
            return 1.0 / rank
    return 0.0


def serialize_source(rank: int, point) -> dict:
    payload = point.payload or {}
    metadata = payload.get("metadata") or {}
    return {
        "rank": rank,
        "source_ids": source_ids_from_point(point),
        "document_id": metadata.get("document_id"),
        "document_type": metadata.get("document_type"),
        "article": metadata.get("article"),
        "clause": metadata.get("clause"),
        "point": metadata.get("point"),
        "heading_title": metadata.get("heading_title"),
        "vector_score": payload.get("_vector_score"),
        "bm25_score": payload.get("_bm25_score"),
        "rrf_score": payload.get("_rrf_score"),
        "candidate_score": payload.get("_candidate_score"),
        "rerank_score_raw": payload.get("_rerank_score_raw"),
        "rerank_score": payload.get("_rerank_score"),
        "text_preview": str(payload.get("page_content") or "")[:1000],
    }


def ask_llm_without_rag(question: str) -> str:
    client = get_llm_client()
    response = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.1,
        messages=[
            {"role": "system", "content": NO_RAG_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    content = response.choices[0].message.content
    return content.strip() if isinstance(content, str) else str(content)


def evaluate_rag_item(item: dict, qdrant_client: QdrantClient, model: SentenceTransformer, top_k: int) -> dict:
    question = item.get("question", "")
    expected_ids = set(map(str, item.get("relevant_context_ids") or []))

    started = time.perf_counter()
    points = retrieve(question, qdrant_client, model, limit=top_k)
    retrieval_seconds = time.perf_counter() - started

    retrieved_ids = source_ids_from_points(points)
    matched_ids = sorted(expected_ids & retrieved_ids)
    recall = len(matched_ids) / len(expected_ids) if expected_ids else 0.0
    mrr = reciprocal_rank(points, expected_ids)

    generation_started = time.perf_counter()
    answer = ask_llm(question, points)
    generation_seconds = time.perf_counter() - generation_started

    return {
        "question_id": item.get("question_id"),
        "question": question,
        "ground_truth_answer": item.get("ground_truth_answer"),
        "doc_name": item.get("doc_name"),
        "expected_context_ids": sorted(expected_ids),
        "retrieved_context_ids": sorted(retrieved_ids),
        "matched_context_ids": matched_ids,
        "hit_at_k": bool(matched_ids),
        "recall_at_k": recall,
        "mrr": mrr,
        "top_k": top_k,
        "retrieval_latency_seconds": retrieval_seconds,
        "generation_latency_seconds": generation_seconds,
        "total_latency_seconds": retrieval_seconds + generation_seconds,
        "answer": answer,
        "answer_char_count": len(answer),
        "sources": [serialize_source(rank, point) for rank, point in enumerate(points, start=1)],
    }


def evaluate_no_rag_item(item: dict) -> dict:
    question = item.get("question", "")
    started = time.perf_counter()
    answer = ask_llm_without_rag(question)
    generation_seconds = time.perf_counter() - started

    return {
        "question_id": item.get("question_id"),
        "question": question,
        "ground_truth_answer": item.get("ground_truth_answer"),
        "doc_name": item.get("doc_name"),
        "generation_latency_seconds": generation_seconds,
        "answer": answer,
        "answer_char_count": len(answer),
    }


def summarize_rag(results: list[dict]) -> dict:
    if not results:
        return {
            "count": 0,
            "hit_at_k": 0.0,
            "mean_recall_at_k": 0.0,
            "mrr": 0.0,
            "mean_total_latency_seconds": 0.0,
        }

    return {
        "count": len(results),
        "hit_at_k": sum(1 for item in results if item["hit_at_k"]) / len(results),
        "mean_recall_at_k": sum(item["recall_at_k"] for item in results) / len(results),
        "mrr": sum(item["mrr"] for item in results) / len(results),
        "mean_retrieval_latency_seconds": sum(item["retrieval_latency_seconds"] for item in results) / len(results),
        "mean_generation_latency_seconds": sum(item["generation_latency_seconds"] for item in results) / len(results),
        "mean_total_latency_seconds": sum(item["total_latency_seconds"] for item in results) / len(results),
    }


def summarize_no_rag(results: list[dict]) -> dict:
    if not results:
        return {"count": 0, "mean_generation_latency_seconds": 0.0}

    return {
        "count": len(results),
        "mean_generation_latency_seconds": sum(item["generation_latency_seconds"] for item in results) / len(results),
    }


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def run_eval(args: argparse.Namespace) -> None:
    dataset = load_dataset(args.dataset)
    selected = dataset[args.offset : args.offset + args.limit]
    if not selected:
        print("Không có mẫu benchmark nào để chạy.")
        return

    output_dir = Path(args.output_dir)
    run_config = {
        "dataset": args.dataset,
        "offset": args.offset,
        "limit": args.limit,
        "top_k": args.top_k,
        "llm_model": LLM_MODEL,
        "embedding_model": MODEL_NAME,
    }

    rag_results = []
    no_rag_results = []

    if not args.no_rag_only:
        qdrant_client = get_qdrant_client()
        model = SentenceTransformer(MODEL_NAME)
        try:
            for index, item in enumerate(selected, start=args.offset + 1):
                print(f"[RAG] {index}: {item.get('question_id')} - {item.get('question')}")
                rag_results.append(evaluate_rag_item(item, qdrant_client, model, args.top_k))
        finally:
            qdrant_client.close()

        rag_payload = {
            "run_config": run_config,
            "summary": summarize_rag(rag_results),
            "results": rag_results,
        }
        rag_path = output_dir / args.rag_output
        write_json(rag_path, rag_payload)
        print(f"Đã lưu RAG results: {rag_path}")
        print(f"RAG summary: {rag_payload['summary']}")

    if not args.rag_only:
        for index, item in enumerate(selected, start=args.offset + 1):
            print(f"[NO_RAG] {index}: {item.get('question_id')} - {item.get('question')}")
            no_rag_results.append(evaluate_no_rag_item(item))

        no_rag_payload = {
            "run_config": run_config,
            "summary": summarize_no_rag(no_rag_results),
            "results": no_rag_results,
        }
        no_rag_path = output_dir / args.no_rag_output
        write_json(no_rag_path, no_rag_payload)
        print(f"Đã lưu no-RAG results: {no_rag_path}")
        print(f"No-RAG summary: {no_rag_payload['summary']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chạy ablation eval: RAG vs hỏi thẳng LLM.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rag-output", default=DEFAULT_RAG_OUTPUT)
    parser.add_argument("--no-rag-output", default=DEFAULT_NO_RAG_OUTPUT)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--rag-only", action="store_true")
    parser.add_argument("--no-rag-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.rag_only and parsed_args.no_rag_only:
        raise SystemExit("Không thể dùng đồng thời --rag-only và --no-rag-only.")
    run_eval(parsed_args)
