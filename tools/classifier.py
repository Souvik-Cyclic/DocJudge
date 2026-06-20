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
