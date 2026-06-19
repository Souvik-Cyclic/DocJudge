"""Table detection — pdfplumber primary, PyMuPDF fallback.

pdfplumber misses tables in some PDFs (returns 0). PyMuPDF's `find_tables()`
uses a different detection strategy and often catches those, so we fall back to
it per-page whenever pdfplumber finds nothing.
"""
from __future__ import annotations

import fitz
