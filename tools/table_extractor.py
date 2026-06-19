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
