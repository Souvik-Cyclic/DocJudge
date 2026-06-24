"""Agent 3: Structurer Agent (type-aware chunker + indexer).

Per document:
  1. classify it (policy / financial / general)  -> tools.classifier
  2. route to the chunking strategy tuned for that type -> tools.chunkers
  3. label chunks (section / content_type / topic_tags) and generate a
     "context_prefix" that situates each chunk (contextual retrieval)
  4. embed (BGE-small) + store in ChromaDB with doc_type metadata.

The type-aware routing means a legal clause, a financial table, and ordinary
prose are no longer cut by the same fixed-size window.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from config import config
from llm import invoke_structured
from models import IndexedChunk
from observability.logging_config import log_info, timed_node
from prompts.structurer_prompt import STRUCTURER_SYSTEM
from tools import chromadb_tool
from tools.chunkers import get_chunker
from tools.classifier import classify_document
from tools.table_extractor import table_to_markdown

class ChunkLabel(BaseModel):
    section: str = "General"
    content_type: Literal["prose", "table", "list"] = "prose"
    topic_tags: list[str] = Field(default_factory=list)
    context_prefix: str = ""

class BatchLabels(BaseModel):
    """Labels for a batch of chunks, in the same order they were given."""

    labels: list[ChunkLabel] = Field(default_factory=list)

_BATCH_SIZE = 10

def _doc_sample(pages: list[dict], max_chars: int = 3000) -> str:
    """Concatenate the first pages' text as a classification sample."""
    parts: list[str] = []
    total = 0
    for page in pages:
        for block in page.get("text_blocks", []):
            parts.append(block)
            total += len(block)
            if total >= max_chars:
                return "\n".join(parts)[:max_chars]
    return "\n".join(parts)[:max_chars]

@timed_node("structurer")
def structurer_node(state: dict) -> dict:
    pages = state.get("extracted_pages", []) or []
    if not pages:
        return {
            "indexed_chunks_count": chromadb_tool.count(),
            "index_status": "no_new_documents",
            "current_step": "structurer_skipped",
        }

    by_doc: dict[str, list[dict]] = {}
    for page in pages:
        by_doc.setdefault(page["doc_name"], []).append(page)

    prose_specs: list[tuple[str, int, int, str, str, str, bool]] = []
    table_chunks: list[dict] = []
    doc_types: dict[str, str] = {}

    for doc_name, dpages in by_doc.items():
        doc_type, confidence, signals = classify_document(doc_name, _doc_sample(dpages))
        doc_types[doc_name] = doc_type
        log_info("structurer",
                 f"{doc_name}: type={doc_type} conf={confidence} signals={signals}")
        chunker = get_chunker(doc_type)

        for page in dpages:
            page_no = page["page_number"]
            from_ocr = bool(page.get("has_ocr_content"))
            for ci, (section, text) in enumerate(chunker(page["text_blocks"])):
                prose_specs.append((doc_name, page_no, ci, text, section, doc_type, from_ocr))
            for ti, table in enumerate(page["tables"]):
                md = table_to_markdown(table)
                if not md:
                    continue
                table_chunks.append(IndexedChunk(
                    chunk_id=f"{doc_name}_p{page_no}_t{ti}",
                    doc_name=doc_name, page=page_no, section="Table",
                    content_type="table", doc_type=doc_type, from_ocr=from_ocr,
                    text=md, metadata={"topic_tags": "table"},
                ).model_dump())

    labels: list[ChunkLabel] = []
    for start in range(0, len(prose_specs), _BATCH_SIZE):
        batch = prose_specs[start : start + _BATCH_SIZE]
        labels.extend(_label_batch(batch))

    indexed: list[dict] = []
    for (doc_name, page_no, ci, text, detected, doc_type, from_ocr), label in zip(prose_specs, labels):
        section = detected if detected and detected != "General" else label.section
        prefix = (label.context_prefix or "").strip()
        body = f"[Context: {prefix}]\n{text}" if prefix else text
        indexed.append(IndexedChunk(
            chunk_id=f"{doc_name}_p{page_no}_c{ci}",
            doc_name=doc_name, page=page_no,
            section=section, content_type=label.content_type, doc_type=doc_type,
            from_ocr=from_ocr,
            text=body, metadata={"topic_tags": ",".join(label.topic_tags)},
        ).model_dump())
    indexed.extend(table_chunks)

    stored = chromadb_tool.add_chunks(indexed)
    if stored == 0:
        return {
            "indexed_chunks_count": 0,
            "index_status": "empty",
            "documents_ingested": chromadb_tool.count() > 0,
            "error": "No chunks were produced from the document(s) — nothing to index.",
            "current_step": "indexed 0 chunks",
        }
    type_summary = ", ".join(f"{d}={t}" for d, t in doc_types.items())
    return {
        "indexed_chunks_count": stored,
        "index_status": "indexed",
        "documents_ingested": True,
        "doc_types": doc_types,
        "current_step": f"indexed {stored} chunks ({type_summary})",
    }

def _label_batch(batch: list[tuple[str, int, int, str, str, str]]) -> list[ChunkLabel]:
    """Label a batch of chunks in a single LLM call.

    Returns one ChunkLabel per input chunk (padded/truncated to match length so
    a miscounting model can never misalign metadata).
    """
    n = len(batch)
    if n == 0:
        return []

    listing = "\n\n".join(
        f"[CHUNK {i}] (doc={spec[0]}, page={spec[1]}, section={spec[4]}, "
        f"type={spec[5]})\n{spec[3][:1000]}"
        for i, spec in enumerate(batch)
    )
    user_msg = (
        f"Label each of the following {n} chunks. Return a `labels` array with "
        f"EXACTLY {n} entries, in the same order as the chunks.\n\n{listing}"
    )
    try:
        result: BatchLabels = invoke_structured(
            config.LLM_MODEL,
            BatchLabels,
            [("system", STRUCTURER_SYSTEM), ("user", user_msg)],
        )
        labels = list(result.labels)
    except Exception:
        labels = []

    if len(labels) < n:
        labels += [ChunkLabel() for _ in range(n - len(labels))]
    return labels[:n]
