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
