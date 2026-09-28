from __future__ import annotations

from functools import lru_cache
from math import exp

from typing import TYPE_CHECKING

from .config import RERANKER_MODEL

if TYPE_CHECKING:
    from FlagEmbedding import FlagReranker


MAX_PASSAGE_CHARS = 1500
HEAD_CHARS = 1000
TAIL_CHARS = 500


@lru_cache(maxsize=1)
def get_reranker() -> FlagReranker:
    from FlagEmbedding import FlagReranker
    return FlagReranker(RERANKER_MODEL, use_fp16=True)


def clean_text(text: str) -> str:
    return " ".join(str(text).split())


def build_metadata_context(metadata: dict) -> str:
    parts = []

    chapter = metadata.get("chapter")
    if chapter:
        parts.append(str(chapter))

    article = metadata.get("article")
    article_title = metadata.get("article_title")
    if article:
        dieu = f"Điều {article}"
        if article_title:
            dieu = f"{dieu}. {article_title}"
        parts.append(dieu)

    clause = metadata.get("clause")
    if clause:
        parts.append(f"Khoản {clause}")

    point = metadata.get("point")
    if point:
        parts.append(f"Điểm {point})")

    return " - ".join(parts)


def truncate_passage(passage: str) -> str:
    passage = clean_text(passage)
    if len(passage) <= MAX_PASSAGE_CHARS:
        return passage

    trong_tam_index = passage.lower().find("trọng tâm:")
    if trong_tam_index != -1:
        head = passage[:HEAD_CHARS]
        focus = passage[trong_tam_index : trong_tam_index + TAIL_CHARS]
        return clean_text(f"{head} ... {focus}")[:MAX_PASSAGE_CHARS]

    return clean_text(f"{passage[:HEAD_CHARS]} ... {passage[-TAIL_CHARS:]}")[:MAX_PASSAGE_CHARS]


def build_passage(payload: dict) -> str:
    metadata = payload.get("metadata") or {}

    if payload.get("page_content"):
        context = build_metadata_context(metadata)
        passage = str(payload["page_content"])
        if context and context not in passage:
            passage = f"{context}\n{passage}"
        return truncate_passage(passage)

    if payload.get("văn_bản_embed"):
        return truncate_passage(str(payload["văn_bản_embed"]))

    passage = (
        f"Ngữ cảnh: {payload.get('ngữ_cảnh_đầy_đủ', '-')}\n"
        f"Nội dung: {payload.get('nội_dung', '-')}"
    )
    return truncate_passage(passage)


def display_score(raw_score: float) -> float:
    return 100.0 / (1.0 + exp(-raw_score))


def rerank_points(question: str, points, top_k: int = 5):
    pairs = []
    valid_points = []

    for point in points:
        payload = point.payload or {}
        passage = build_passage(payload)
        if not passage.strip():
            continue
        pairs.append([question, passage])
        valid_points.append(point)

    if not pairs:
        return []

    scores = get_reranker().compute_score(pairs, normalize=False)
    if isinstance(scores, float):
        scores = [scores]

    reranked = sorted(
        zip(valid_points, scores),
        key=lambda item: item[1],
        reverse=True,
    )

    top_points = []
    for point, rerank_score in reranked[:top_k]:
        raw_score = float(rerank_score)
        payload = point.payload or {}
        payload.setdefault("_vector_score", float(point.score or 0.0))
        payload["_candidate_score"] = float(point.score or 0.0)
        payload["_rerank_score_raw"] = raw_score
        payload["_rerank_score"] = display_score(raw_score)
        point.payload = payload
        top_points.append(point)

    return top_points
