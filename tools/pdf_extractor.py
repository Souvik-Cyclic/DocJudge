"""PyMuPDF (fitz) wrapper — primary text + layout extraction."""
from __future__ import annotations

import logging
import os

import fitz

for _noisy in ("pdfminer", "pdfminer.pdffont", "pdfminer.pdfinterp", "pdfplumber"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

def extract_text_blocks(pdf_path: str) -> list[dict]:
    """Return per-page text blocks.

    Output: [{"page": int, "blocks": [str, ...], "raw_len": int}, ...]
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)

    pages: list[dict] = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            blocks = page.get_text("blocks")
            texts = [b[4].strip() for b in blocks if b[4].strip()]
            full = "\n".join(texts)
            pages.append({"page": i, "blocks": texts, "raw_len": len(full)})
    return pages

def page_is_empty(page_record: dict, threshold: int = 20) -> bool:
    """Heuristic: page yielded almost no text -> likely scanned -> needs OCR."""
    return page_record.get("raw_len", 0) < threshold
