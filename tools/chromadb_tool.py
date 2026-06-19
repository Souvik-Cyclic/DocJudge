"""ChromaDB client wrapper — persistent vector store with metadata filtering.

We pass our own BGE embeddings (embedding_function=None) so retrieval matches
the embeddings used at index time.
"""
from __future__ import annotations

from functools import lru_cache

import chromadb

from config import config
from tools.embedder import embed_query, embed_texts

@lru_cache(maxsize=1)
def _client():
    return chromadb.PersistentClient(path=config.CHROMA_DIR)
