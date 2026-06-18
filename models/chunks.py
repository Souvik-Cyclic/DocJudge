"""Output schema for the Structurer Agent (Agent 3)."""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

class IndexedChunk(BaseModel):
    """A semantically coherent chunk ready for embedding + ChromaDB storage."""

    chunk_id: str
    doc_name: str
    page: int
    section: str = "unknown"
    content_type: Literal["prose", "table", "list"] = "prose"
    doc_type: str = "general"
    from_ocr: bool = False
    text: str
    metadata: dict = Field(default_factory=dict)
