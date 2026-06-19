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

def ocr_status() -> dict:
    """Diagnostic for the UI / logs: is OCR usable, and why not if not."""
    if not _HAS_OCR:
        return {"available": False, "engine": None,
                "reason": "rapidocr-onnxruntime not installed "
                          "(pip install rapidocr-onnxruntime)"}
    if not ocr_available():
        return {"available": False, "engine": "rapidocr",
                "reason": "RapidOCR failed to initialize"}
    return {"available": True, "engine": "rapidocr", "reason": "ok"}

def ocr_page(pdf_path: str, page_number: int, dpi: int = 200) -> str:
    """OCR a single (1-indexed) page. Returns extracted text, or "" if unavailable."""
    engine = _get_engine()
    if engine is None:
        return ""
    with fitz.open(pdf_path) as doc:
        page = doc[page_number - 1]
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
    return _ocr_pil(img)

def ocr_image(image_path: str) -> str:
    """OCR a standalone image file (png/jpg/...). Returns text, or "" if unavailable."""
    engine = _get_engine()
    if engine is None:
        return ""
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return ""
    return _ocr_pil(img)

def _ocr_pil(img) -> str:
    engine = _get_engine()
    if engine is None:
        return ""
    result, _elapse = engine(np.array(img))
    if not result:
        return ""
    return "\n".join(line[1] for line in result).strip()
