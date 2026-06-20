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

_WORD = re.compile(r"[a-z0-9][a-z0-9\-]+")

def _signal_score(text: str, signals: tuple[str, ...]) -> tuple[float, list[str]]:
    """Count signal-word hits, normalized by sample length. Returns (score, hits)."""
    hits = []
    total = 0
    for sig in signals:
        c = text.count(sig)
        if c:
            total += c
            hits.append(sig)
    score = total / (max(len(text), 1) / 1000.0)
    return score, hits

def _numeric_density(text: str) -> float:
    if not text:
        return 0.0
    digits = sum(ch.isdigit() for ch in text)
    return digits / len(text)

def classify_heuristic(sample_text: str) -> tuple[DocType, float, list[str]]:
    """Deterministic, free classification. Returns (doc_type, confidence, signals)."""
    low = sample_text.lower()
    fin_score, fin_hits = _signal_score(low, _FINANCIAL_SIGNALS)
    pol_score, pol_hits = _signal_score(low, _POLICY_SIGNALS)

    fin_score += _numeric_density(low) * 8.0

    best = max(fin_score, pol_score)
    if best < 0.5:
        return "general", 0.3, []

    if fin_score >= pol_score:
        doc_type: DocType = "financial"
        signals = fin_hits
    else:
        doc_type = "policy"
        signals = pol_hits

    margin = abs(fin_score - pol_score) / (best or 1.0)
    confidence = round(min(1.0, 0.4 + margin), 2)
    return doc_type, confidence, signals[:6]

def _classify_llm(doc_name: str, sample_text: str) -> DocType | None:
    """Best-effort LLM tie-breaker. Returns a doc_type or None on any failure."""
    try:
        from pydantic import BaseModel

        from config import config
        from llm import invoke_structured

        class _Guess(BaseModel):
            doc_type: DocType = "general"
            reason: str = ""

        system = (
            "Classify the document excerpt as exactly one of: 'policy' (legal / "
            "policy / governance / terms with clauses and obligations), "
            "'financial' (financial statements, reports, figures, tables of "
            "numbers), or 'general'. Return only the type and a one-line reason."
        )
        user = f"Document name: {doc_name}\n\nExcerpt:\n{sample_text[:2500]}"
        guess: _Guess = invoke_structured(
            config.LLM_MODEL, _Guess, [("system", system), ("user", user)]
        )
        return guess.doc_type
    except Exception:
        return None

def classify_document(doc_name: str, sample_text: str) -> tuple[DocType, float, list[str]]:
    """Classify a document. Heuristic first; LLM only breaks low-confidence ties.

    Always returns a usable (doc_type, confidence, signals) — never raises.
    """
    doc_type, confidence, signals = classify_heuristic(sample_text)
    if confidence < 0.55:
        llm_type = _classify_llm(doc_name, sample_text)
        if llm_type is not None:
            return llm_type, max(confidence, 0.55), signals
    return doc_type, confidence, signals
