"""Agent 2: Extraction Agent.

Content extraction from many file types (PDF, images, Word, PowerPoint,
spreadsheets, text) via a format dispatcher. Each source is normalized into
ExtractedPage records (text blocks + tables) for the Structurer. Scanned PDFs
and images fall back to RapidOCR. No file-size or page limit.
"""
from __future__ import annotations

import os

from models import ExtractedPage
from observability.logging_config import log_info, timed_node
from tools import ocr_tool
from tools.file_extractors import extract_any

def extract_file(path: str) -> list[ExtractedPage]:
    doc_name = os.path.basename(path)
    out: list[ExtractedPage] = []
    for rec in extract_any(path):
        out.append(ExtractedPage(
            doc_name=doc_name,
            page_number=rec.get("page", 1),
            text_blocks=rec.get("blocks", []),
            tables=rec.get("tables", []),
            has_ocr_content=bool(rec.get("ocr")),
        ))
    return out

def _page_has_content(page: dict) -> bool:
    text = "".join(page.get("text_blocks", [])).strip()
    return bool(text) or bool(page.get("tables"))

@timed_node("extraction")
def extraction_node(state: dict) -> dict:
    docs = state.get("documents", []) or []
    if not docs:
        return {"extracted_pages": [], "current_step": "extraction_skipped"}

    all_pages: list[dict] = []
    empty_docs: list[str] = []
    empty_pages: dict[str, int] = {}
    for path in docs:
        if not os.path.exists(path):
            empty_docs.append(f"{os.path.basename(path)} (not found)")
            continue
        try:
            pages = extract_file(path)
        except Exception as exc:
            log_info("extraction", f"failed to read {path}: {exc}")
            empty_docs.append(f"{os.path.basename(path)} (unreadable: {exc})")
            continue
        page_dicts = [p.model_dump() for p in pages]
        if not any(_page_has_content(p) for p in page_dicts):
            empty_docs.append(
                f"{os.path.basename(path)} (no extractable text - scanned image? "
                f"install rapidocr-onnxruntime)")
            log_info("extraction", f"WARNING: {path} produced no extractable content")
        for rec in page_dicts:
            if not any(b.strip() for b in rec.get("text_blocks", [])) and not rec.get("tables"):
                empty_pages[rec["doc_name"]] = empty_pages.get(rec["doc_name"], 0) + 1
        all_pages.extend(page_dicts)

    warnings: list[str] = []
    if empty_pages and not ocr_tool.ocr_available():
        status = ocr_tool.ocr_status()
        for doc, n in empty_pages.items():
            warnings.append(
                f"{doc}: {n} page(s) have no extractable text and appear scanned, "
                f"but OCR is unavailable ({status['reason']}). Install "
                f"rapidocr-onnxruntime to index them."
            )
        log_info("extraction", "; ".join(warnings))

    update = {
        "extracted_pages": all_pages,
        "extraction_warnings": warnings,
        "current_step": f"extracted {len(all_pages)} pages from {len(docs)} file(s)",
    }
    if empty_docs:
        update["error"] = "Some documents had no extractable content: " + "; ".join(empty_docs)
    return update
