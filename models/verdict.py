"""Output schema for the Judge Agent (Agent 5)."""
from __future__ import annotations

from typing import Optional, Union

from pydantic import BaseModel, Field, field_validator

def _to_bool(v):
    if isinstance(v, bool) or v is None:
        return v
    s = str(v).strip().lower()
    if s in {"true", "yes", "1"}:
        return True
    if s in {"false", "no", "0"}:
        return False
    return v

class JudgeVerdict(BaseModel):
    """Independent verification verdict over an AnswerResponse."""

    grounded: Union[bool, str] = False
    hallucination_detected: Union[bool, str] = False
    citations_valid: Union[bool, str] = False
    refusal_appropriate: Optional[Union[bool, str]] = None
    issues: list[str] = Field(default_factory=list)
    overall_pass: Union[bool, str] = False
    feedback_for_retry: Optional[str] = None

    @field_validator(
        "grounded", "hallucination_detected", "citations_valid",
        "refusal_appropriate", "overall_pass", mode="before",
    )
    @classmethod
    def _coerce_bool(cls, v):
        return _to_bool(v)
