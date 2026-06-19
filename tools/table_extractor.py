"""Table detection — pdfplumber primary, PyMuPDF fallback.

pdfplumber misses tables in some PDFs (returns 0). PyMuPDF's `find_tables()`
uses a different detection strategy and often catches those, so we fall back to
it per-page whenever pdfplumber finds nothing.
"""
from __future__ import annotations

import fitz
import pdfplumber

def _clean(table: list) -> list[list[str]]:
    return [[(cell or "").strip() for cell in row] for row in table if row]

def _is_real_table(table: list[list[str]]) -> bool:
    """Reject degenerate 'tables' (single column / single row / all-empty) that
    are really just lines of text picked up as a one-column grid."""
    if not table or len(table) < 2:
        return False
    max_cols = max(len(r) for r in table)
    if max_cols < 2:
        return False
    non_empty = sum(1 for r in table for c in r if c.strip())
    return non_empty >= 2

def _pymupdf_tables(pdf_path: str, page_index0: int) -> list[list[list[str]]]:
    """PyMuPDF table extraction for a single 0-indexed page."""
    out: list[list[list[str]]] = []
    try:
        with fitz.open(pdf_path) as doc:
            page = doc[page_index0]
            finder = page.find_tables()
            for tbl in finder.tables:
                rows = tbl.extract()
                cleaned = _clean(rows)
                if cleaned:
                    out.append(cleaned)
    except Exception:
        pass
    return out

def extract_tables(pdf_path: str) -> dict[int, list[list[list[str]]]]:
    """Return {page_number(1-indexed): [table, ...]}, table = list[rows]=list[cells].

    pdfplumber first; if a page yields no tables, retry that page with PyMuPDF.
    """
    out: dict[int, list[list[list[str]]]] = {}
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            tables = page.extract_tables() or []
            cleaned = [_clean(t) for t in tables if t]
            cleaned = [t for t in cleaned if _is_real_table(t)]
            if not cleaned:
                cleaned = [t for t in _pymupdf_tables(pdf_path, i - 1)
                           if _is_real_table(t)]
            if cleaned:
                out[i] = cleaned
    return out

def table_to_markdown(table: list[list[str]]) -> str:
    """Render a table as markdown so the LLM keeps row/column structure."""
    if not table:
        return ""
    header, *rows = table
    md = ["| " + " | ".join(header) + " |"]
    md.append("| " + " | ".join("---" for _ in header) + " |")
    for row in rows:
        row = row + [""] * (len(header) - len(row))
        md.append("| " + " | ".join(row[: len(header)]) + " |")
    return "\n".join(md)
