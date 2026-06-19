"""BGE-small-en-v1.5 embedding wrapper (sentence-transformers, local + free)."""
from __future__ import annotations

from functools import lru_cache

from config import config

@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBED_MODEL)
