from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from traffic_law_rag.config import BENCHMARK_DIR, EMBEDDING_MODEL, QDRANT_PATH
from traffic_law_rag.qdrant import get_qdrant_client
from traffic_law_rag.rag import ask_llm, retrieve

MODEL_NAME = EMBEDDING_MODEL


DEFAULT_DATASET = str(BENCHMARK_DIR / "generated_rag_dataset.json")


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def load_dataset(path: str) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Benchmark dataset phải là một JSON list.")

    return data


def source_ids_from_points(points) -> set[str]:
    source_ids = set()
    for point in points:
        payload = point.payload or {}
        metadata = payload.get("metadata") or {}

        for value in [
            payload.get("chunk_id"),
            payload.get("source"),
            metadata.get("source"),
        ]:
            if value:
                source_ids.add(str(value))

        for source_id in payload.get("_source_chunk_ids") or []:
            source_ids.add(str(source_id))

    return source_ids


def run_benchmark(
    dataset_path: str,
    limit: int,
    offset: int,
    top_k: int,
    with_llm: bool,
) -> None:
    dataset = load_dataset(dataset_path)
    selected = dataset[offset : offset + limit]
    if not selected:
        print("Không có mẫu benchmark nào để chạy.")
        return

    qdrant_client = get_qdrant_client()
    model = SentenceTransformer(MODEL_NAME)

    hits = 0
    try:
        for index, item in enumerate(selected, start=offset + 1):
            question = item.get("question", "")
            expected_ids = set(map(str, item.get("relevant_context_ids") or []))

            print("\n" + "=" * 80)
            print(f"[{index}] {item.get('question_id', '-')}")
            print(f"QUESTION: {question}")
            print(f"EXPECTED: {', '.join(sorted(expected_ids)) or '-'}")

            points = retrieve(question, qdrant_client, model, limit=top_k)
            retrieved_ids = source_ids_from_points(points)
            matched_ids = expected_ids & retrieved_ids
            is_hit = bool(matched_ids)
            hits += int(is_hit)

            print(f"HIT@{top_k}: {'YES' if is_hit else 'NO'}")
            print(f"MATCHED: {', '.join(sorted(matched_ids)) or '-'}")
            print("TOP SOURCES:")
            for rank, point in enumerate(points, start=1):
                payload = point.payload or {}
                metadata = payload.get("metadata") or {}
                source_ids = payload.get("_source_chunk_ids") or [payload.get("chunk_id", "-")]
                citation = (
                    f"{metadata.get('document_type', 'Văn bản')} {metadata.get('document_id', '-')} "
                    f"Điều {metadata.get('article', '-')} "
                    f"Khoản {metadata.get('clause', '-')} "
                    f"Điểm {metadata.get('point', '-')}"
                )
                print(f"  {rank}. {citation} | {', '.join(map(str, source_ids))}")

            if with_llm:
                answer = ask_llm(question, points)
                print("ANSWER PREVIEW:")
                print(answer[:1200])

        print("\n" + "=" * 80)
        print(f"SUMMARY: hit@{top_k} = {hits}/{len(selected)} = {hits / len(selected):.2%}")
    finally:
        qdrant_client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chạy thử benchmark RAG trên vài câu hỏi.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Gọi LLM để sinh câu trả lời. Mặc định chỉ test retrieval.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        dataset_path=args.dataset,
        limit=args.limit,
        offset=args.offset,
        top_k=args.top_k,
        with_llm=args.with_llm,
    )
