"""RapidOCR fallback — only fires when a page yields little/no native text.

RapidOCR is fully pip-installable (ONNX models ship with the package), so there
is NO system binary to install and no PATH setup:

    pip install rapidocr-onnxruntime

It runs locally and free. We render each scanned page to an image with PyMuPDF
and OCR it with RapidOCR.
"""
from __future__ import annotations

import io

import fitz
import numpy as np
from PIL import Image

try:
    from rapidocr_onnxruntime import RapidOCR

    _HAS_OCR = True
except Exception:
    _HAS_OCR = False

_engine = None

def _get_engine():
    global _engine
    if _engine is None and _HAS_OCR:
        _engine = RapidOCR()
    return _engine

def ocr_available() -> bool:
    if not _HAS_OCR:
        return False
    try:
        return _get_engine() is not None
    except Exception:
        return False
