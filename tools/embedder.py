"""BGE-small-en-v1.5 embedding wrapper (sentence-transformers, local + free)."""
from __future__ import annotations

from functools import lru_cache

from config import config

@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(config.EMBED_MODEL)

def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a batch of texts. BGE recommends normalized embeddings for cosine."""
    model = _model()
    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [v.tolist() for v in vecs]

def embed_query(text: str) -> list[float]:
    """BGE retrieval works best with an instruction prefix on the query."""
    prefixed = f"Represent this sentence for searching relevant passages: {text}"
    return embed_texts([prefixed])[0]
