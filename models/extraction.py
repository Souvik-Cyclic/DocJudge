"""Output schema for the Extraction Agent (Agent 2)."""
from __future__ import annotations

from pydantic import BaseModel, Field

class ExtractedPage(BaseModel):
    """Raw content extracted from a single PDF page."""

    doc_name: str
    page_number: int
    text_blocks: list[str] = Field(default_factory=list)
    tables: list[list[list[str]]] = Field(default_factory=list)
    has_ocr_content: bool = False
