"""Qdrant client factory supporting local persistent storage."""

from __future__ import annotations

from qdrant_client import QdrantClient

from .config import QDRANT_PATH


def get_qdrant_client() -> QdrantClient:
    QDRANT_PATH.parent.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(QDRANT_PATH))
