"""Pydantic schemas for every agent handoff (structured outputs)."""
from .extraction import ExtractedPage
from .chunks import IndexedChunk
from .answer import AnswerResponse, CitedChunk
from .verdict import JudgeVerdict

__all__ = [
    "ExtractedPage",
    "IndexedChunk",
    "AnswerResponse",
    "CitedChunk",
    "JudgeVerdict",
]
