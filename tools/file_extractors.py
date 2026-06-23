"""Multi-format extractors -> normalized (text_blocks, tables) per "page".

Supported: PDF, images (RapidOCR), Word (.docx), PowerPoint (.pptx),
spreadsheets (.xlsx/.xls/.csv), plain text (.txt/.md).

Each extractor returns a list of "page" dicts:
    {"page": int, "blocks": [str, ...], "tables": [[[cell,...],...], ...],
     "ocr": bool}
so the extraction agent can treat every format uniformly.
"""
from __future__ import annotations

import os

from tools import ocr_tool
from tools.pdf_extractor import extract_text_blocks, page_is_empty
from tools.table_extractor import extract_tables

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp", ".gif"}
DOC_EXTS = {".docx"}
PPT_EXTS = {".pptx"}
SHEET_EXTS = {".xlsx", ".xls", ".csv"}
