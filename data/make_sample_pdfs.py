"""Generate sample policy/finance PDFs (incl. a table) for demo + evaluation.

Run from repo root:
    python -m data.make_sample_pdfs

Produces:
    data/sample_pdfs/annual_report_2025.pdf   (prose + a financial table)
    data/sample_pdfs/risk_policy.pdf          (prose, risk factors)
"""
from __future__ import annotations

import os

import fitz

OUT_DIR = "data/sample_pdfs"
