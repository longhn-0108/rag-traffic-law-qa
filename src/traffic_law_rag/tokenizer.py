#!/usr/bin/env python3
"""Utility module for tokenization using the BGE-M3 tokenizer.

Provides a cached token-counting function used by the chunking pipeline
to enforce per-chunk token limits.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any

_tokenizer: Any = None


def get_tokenizer() -> Any:
    """Lazily load and cache the BGE-M3 tokenizer."""
    global _tokenizer
    if _tokenizer is None:
        try:
            from transformers import AutoTokenizer
            _tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        except Exception as exc:
            print(
                f"Warning: Could not load BGE-M3 tokenizer: {exc}. "
                "Using fallback estimator.",
                flush=True,
            )
            _tokenizer = False
    return _tokenizer


@lru_cache(maxsize=30_000)
def count_tokens(text: str) -> int:
    """Count the number of tokens in *text* using the BGE-M3 tokenizer.

    Falls back to a word-count estimator (×1.35) when the tokenizer is
    unavailable.
    """
    if not text:
        return 0

    tokenizer = get_tokenizer()
    if tokenizer:
        try:
            return len(tokenizer.encode(text, add_special_tokens=False))
        except Exception:
            pass

    # Fallback: rough word-based estimation
    return int(len(re.findall(r"\S+", text)) * 1.35)
