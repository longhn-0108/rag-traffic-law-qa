from __future__ import annotations

import argparse
import json
import sys
import uuid
from pathlib import Path

from qdrant_client import QdrantClient
from .config import BATCH_SIZE, CHUNKS_PATH, COLLECTION_NAME, EMBEDDING_MODEL, QDRANT_PATH, PROCESSED_DIR
from qdrant_client.models import Distance, Filter, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


DEFAULT_CHUNKS_PATH = str(CHUNKS_PATH)
DEFAULT_DB_PATH = str(QDRANT_PATH)
VECTOR_SIZE = 1024
MODEL_NAME = EMBEDDING_MODEL


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")


def load_chunks(chunks_path: str) -> list[dict]:
    path = Path(chunks_path)
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy file chunks: {path}")

    with path.open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not isinstance(chunks, list):
        raise ValueError("semantic_chunks.json phải là một JSON list.")

    valid_chunks = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue

        chunk_id = chunk.get("id")
        embed_text = chunk.get("embed_text")
        if not chunk_id or not embed_text:
            continue

        valid_chunks.append(chunk)

    if not valid_chunks:
        raise ValueError("Không có chunk hợp lệ nào có đủ id và embed_text.")

    return valid_chunks


def build_payload(chunk: dict) -> dict:
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


def prepare_collection(client: QdrantClient, collection_name: str, recreate: bool) -> None:
    if client.collection_exists(collection_name):
        if recreate:
            print(f"--- Xóa toàn bộ points cũ trong collection: {collection_name} ---")
            client.delete(
                collection_name=collection_name,
                points_selector=Filter(must=[]),
                wait=True,
            )
        return

    print(f"--- Tạo collection mới: {collection_name} ---")
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )


def run_ingestion(
    chunks_path: str = DEFAULT_CHUNKS_PATH,
    db_path: str = DEFAULT_DB_PATH,
    collection_name: str = COLLECTION_NAME,
    recreate: bool = True,
) -> None:
    print(f"--- Đọc chunks từ: {chunks_path} ---")
    chunks = load_chunks(chunks_path)
    print(f"--- Số chunks hợp lệ: {len(chunks)} ---")

    print(f"--- Tải embedding model: {MODEL_NAME} ---")
    model = SentenceTransformer(MODEL_NAME)

    client = QdrantClient(path=db_path)
    try:
        prepare_collection(client, collection_name, recreate=recreate)

        print(f"--- Embedding và upsert vào Qdrant: {db_path} ---")
        for i in tqdm(range(0, len(chunks), BATCH_SIZE)):
            batch = chunks[i : i + BATCH_SIZE]
            texts = [str(chunk["embed_text"]) for chunk in batch]
            embeddings = model.encode(texts, normalize_embeddings=True)

            points = []
            for chunk, embedding in zip(batch, embeddings):
                chunk_id = str(chunk["id"])
                point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=embedding.tolist(),
                        payload=build_payload(chunk),
                    )
                )

            client.upsert(collection_name=collection_name, points=points)

        print(f"--- HOÀN THÀNH: {len(chunks)} chunks đã được lưu vào '{collection_name}' ---")
    finally:
        client.close()


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest semantic_chunks.json vào Qdrant local."
    )
    parser.add_argument("--chunks", default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--db", default=DEFAULT_DB_PATH)
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Không xóa collection cũ trước khi upsert.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    run_ingestion(
        chunks_path=args.chunks,
        db_path=args.db,
        collection_name=args.collection,
        recreate=not args.append,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
