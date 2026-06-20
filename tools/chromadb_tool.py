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

def get_collection():
    return _client().get_or_create_collection(
        name=config.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

def add_chunks(chunks: list[dict]) -> int:
    """Store IndexedChunk dicts. Returns number stored.

    Each chunk dict needs: chunk_id, text, doc_name, page, section, content_type.
    """
    if not chunks:
        return 0
    col = get_collection()
    ids = [c["chunk_id"] for c in chunks]
    docs = [c["text"] for c in chunks]
    metas = [
        {
            "doc_name": c["doc_name"],
            "page": c["page"],
            "section": c.get("section", "unknown"),
            "content_type": c.get("content_type", "prose"),
            "doc_type": c.get("doc_type", "general"),
            "from_ocr": bool(c.get("from_ocr", False)),
        }
        for c in chunks
    ]
    embeddings = embed_texts(docs)
    col.upsert(ids=ids, documents=docs, metadatas=metas, embeddings=embeddings)
    return len(ids)

def query(question: str, top_k: int = 10, where: dict | None = None) -> list[dict]:
    """Retrieve top-k chunks. `where` is an optional Chroma metadata filter.

    Returns [{chunk_id, text, doc_name, page, section, content_type, distance}, ...]
    """
    col = get_collection()
    if col.count() == 0:
        return []
    q_emb = embed_query(question)
    res = col.query(
        query_embeddings=[q_emb],
        n_results=min(top_k, col.count()),
        where=where or None,
    )
    out: list[dict] = []
    ids = res["ids"][0]
    docs = res["documents"][0]
    metas = res["metadatas"][0]
    dists = res["distances"][0]
    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        out.append({"chunk_id": cid, "text": doc, "distance": dist, **meta})
    return out

def get_doc_chunks(doc_names: list[str] | None = None) -> list[dict]:
    """Return all stored chunks (text + metadata) for inspection in the UI.

    If doc_names is given, only those documents are returned. Sorted by
    (doc_name, page, chunk_id) so the index reads in document order.
    """
    col = get_collection()
    if col.count() == 0:
        return []
    where = None
    if doc_names:
        where = ({"doc_name": {"$eq": doc_names[0]}} if len(doc_names) == 1
                 else {"doc_name": {"$in": list(doc_names)}})
    got = col.get(where=where, include=["documents", "metadatas"])
    out: list[dict] = []
    for cid, doc, meta in zip(got["ids"], got["documents"], got["metadatas"]):
        out.append({"chunk_id": cid, "text": doc, **meta})
    out.sort(key=lambda c: (c.get("doc_name", ""), c.get("page", 0), c.get("chunk_id", "")))
    return out
