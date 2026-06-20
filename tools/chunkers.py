"""Document-type-aware chunking strategies.

Each chunker takes a page's text blocks and returns a list of
``(section_name, chunk_text)`` tuples. They share one section-aware packer but
differ in heading detection, target size, overlap, and how they treat
number-dense blocks — because a legal clause, a financial table, and ordinary
prose should not be cut the same way.

Strategies:
  - ``default_chunker``   — generic prose (the original behavior).
  - ``policy_chunker``    — clause/section aware; smaller, tighter chunks that
                            respect legal hierarchy (Article / Section / clause).
  - ``financial_chunker`` — keeps number/table-dense blocks intact so a figure
                            never gets split from its row/header context.

Pick one with ``get_chunker(doc_type)``.
"""
from __future__ import annotations

import re
from typing import Callable

_HEADING_RE = re.compile(
    r"^\s*("
    r"\d+(\.\d+)*[.)]?\s+[A-Z]"
    r"|[A-Z][A-Za-z0-9 ,&/()-]{2,60}:"
    r")"
)
_LEGAL_RE = re.compile(r"^\s*(ARTICLE|SECTION|CLAUSE|SCHEDULE|EXHIBIT)\s+[IVXLC\d]",
                       re.IGNORECASE)

def _first_line(block: str) -> str:
    s = block.strip()
    return s.splitlines()[0].strip() if s else ""

def _is_heading_generic(block: str) -> bool:
    """Heuristic: short line that looks like a section header."""
    line = _first_line(block)
    if not line or len(line) > 80:
        return False
    if _HEADING_RE.match(line):
        return True
    words = line.split()
    if 1 <= len(words) <= 8 and not line.endswith((".", "?", "!")):
        caps = sum(1 for w in words if w[:1].isupper())
        if caps >= max(1, len(words) - 1):
            return True
    return False

def _is_heading_legal(block: str) -> bool:
    """Generic headings plus legal numbering and short ALL-CAPS headers."""
    line = _first_line(block)
    if not line:
        return False
    if _LEGAL_RE.match(line):
        return True
    if 3 < len(line) <= 70 and line.isupper() and any(c.isalpha() for c in line):
        return True
    return _is_heading_generic(block)
