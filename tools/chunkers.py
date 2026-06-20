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

def _safe_tail(text: str, max_chars: int) -> str:
    """Last <=max_chars of text, trimmed to start on a word boundary so overlap
    never cuts a word or a number in half."""
    if len(text) <= max_chars:
        return text
    tail = text[-max_chars:]
    sp = tail.find(" ")
    return tail[sp + 1:] if 0 <= sp < len(tail) - 1 else tail

def _is_numeric_block(block: str) -> bool:
    """True for table-like / figure-dense blocks that must stay intact."""
    s = block.strip()
    if len(s) < 8:
        return False
    digits = sum(ch.isdigit() for ch in s)
    if digits / len(s) > 0.20:
        return True
    return s.count("|") >= 4 or s.count("\t") >= 3

def _group_sections(blocks: list[str],
                    heading_fn: Callable[[str], bool]) -> list[tuple[str, list[str]]]:
    """Group blocks under the nearest preceding heading."""
    sections: list[tuple[str, list[str]]] = []
    current_name = "General"
    current_body: list[str] = []
    for b in blocks:
        if not b.strip():
            continue
        if heading_fn(b):
            if current_body:
                sections.append((current_name, current_body))
                current_body = []
            current_name = _first_line(b)
        else:
            current_body.append(b)
    if current_body:
        sections.append((current_name, current_body))
    if not sections:
        sections = [("General", [b for b in blocks if b.strip()])]
    return sections

def _pack(blocks: list[str], *, target: int, overlap: int,
         heading_fn: Callable[[str], bool],
         isolate_fn: Callable[[str], bool] | None = None) -> list[tuple[str, str]]:
    """Split at headings, then pack each section's body to ~target chars without
    crossing the boundary. ``isolate_fn`` blocks are emitted as their own chunk
    so they keep their local context (used for financial tables/figures).
    """
    out: list[tuple[str, str]] = []
    for name, body in _group_sections(blocks, heading_fn):
        cur = ""
        for b in body:
            if isolate_fn and isolate_fn(b):
                if cur:
                    out.append((name, cur))
                    cur = ""
                out.append((name, b))
                continue
            if len(cur) + len(b) + 1 <= target:
                cur = f"{cur}\n{b}".strip()
            else:
                if cur:
                    out.append((name, cur))
                tail = _safe_tail(cur, overlap) if cur else ""
                cur = f"{tail}\n{b}".strip() if tail else b
        if cur:
            out.append((name, cur))
    return out

def default_chunker(blocks: list[str]) -> list[tuple[str, str]]:
    """Generic prose chunking (original behavior): ~900 chars, 150 overlap."""
    return _pack(blocks, target=900, overlap=150, heading_fn=_is_heading_generic)
