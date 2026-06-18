"""Output schema for the Answer Agent (Agent 4)."""
from __future__ import annotations

from typing import Literal, Union

from pydantic import BaseModel, Field, field_validator

class CitedChunk(BaseModel):
    """A chunk the answer explicitly cites as support."""

    chunk_id: str
    doc_name: str
    page: Union[int, str] = 0
    snippet: str

    @field_validator("page", mode="before")
    @classmethod
    def _coerce_page(cls, v):
        try:
            return int(str(v).strip())
        except (ValueError, AttributeError):
            return 0

class AnswerResponse(BaseModel):
    """Grounded answer with citations and self-reported confidence."""

    answer: str
    cited_chunks: Union[list[CitedChunk], str] = Field(default_factory=list)
    confidence: Literal["high", "medium", "low", "not_found"] = "medium"
    refusal: Union[bool, str] = False

    @field_validator("cited_chunks", mode="before")
    @classmethod
    def _coerce_cited(cls, v):
        if isinstance(v, str):
            import json
            try:
                parsed = json.loads(v)
                return parsed if isinstance(parsed, list) else []
            except Exception:
                return []
        return v

    @field_validator("refusal", mode="before")
    @classmethod
    def _coerce_refusal(cls, v):
        if isinstance(v, bool):
            return v
        s = str(v).strip().lower()
        if s in {"true", "yes", "1"}:
            return True
        if s in {"false", "no", "0", ""}:
            return False
        return bool(v)
