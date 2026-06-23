"""Agent 2: Extraction Agent.

Content extraction from many file types (PDF, images, Word, PowerPoint,
spreadsheets, text) via a format dispatcher. Each source is normalized into
ExtractedPage records (text blocks + tables) for the Structurer. Scanned PDFs
and images fall back to RapidOCR. No file-size or page limit.
"""
from __future__ import annotations

import os

from models import ExtractedPage
