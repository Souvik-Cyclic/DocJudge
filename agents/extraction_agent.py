"""Agent 2: Extraction Agent.

Content extraction from many file types (PDF, images, Word, PowerPoint,
spreadsheets, text) via a format dispatcher. Each source is normalized into
ExtractedPage records (text blocks + tables) for the Structurer. Scanned PDFs
and images fall back to RapidOCR. No file-size or page limit.
"""
from __future__ import annotations

import os

from models import ExtractedPage
from observability.logging_config import log_info, timed_node
from tools import ocr_tool
from tools.file_extractors import extract_any
