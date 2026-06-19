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
