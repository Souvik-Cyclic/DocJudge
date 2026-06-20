"""Document type classifier (free, local-first).

Decides whether an ingested PDF is a *policy/legal* document, a *financial*
document, or *general* prose, so the Structurer can pick a chunking strategy
tuned to that shape (clause-aware vs table/number-aware vs generic).

Design:
  - A deterministic, dependency-free keyword/numeric-density heuristic runs
    first. It is free and never fails.
  - Only when the heuristic is *ambiguous* (the top two scores are close) do we
    fall back to a single best-effort LLM call to break the tie. Any failure
    there silently keeps the heuristic result, so classification never blocks
    ingestion.
"""
from __future__ import annotations

import re
from typing import Literal

DocType = Literal["policy", "financial", "general"]

_FINANCIAL_SIGNALS = (
    "revenue", "ebitda", "income statement", "balance sheet", "cash flow",
    "fiscal", "quarter", "q1", "q2", "q3", "q4", "dividend", "gross margin",
    "operating margin", "net income", "assets", "liabilities", "earnings",
    "shareholder", "gaap", "profit", "expenses", "depreciation", "equity",
    "guidance", "year-over-year", "yoy", "annual report", "statement of",
)
_POLICY_SIGNALS = (
    "policy", "shall", "must", "prohibited", "compliance", "governance",
    "clause", "article", "hereby", "whereas", "definitions", "scope",
    "responsibilities", "procedure", "terms and conditions", "accordance",
    "guidelines", "regulation", "mandatory", "enforcement", "applicable",
    "obligations", "permitted", "violation", "the company shall", "section",
)
