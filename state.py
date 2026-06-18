"""Shared LangGraph state passed across all nodes."""
from __future__ import annotations

import operator
from typing import Annotated, Optional, TypedDict

class GraphState(TypedDict, total=False):
    user_question: str
    documents: list[str]

    execution_plan: str
    current_step: str
    retry_count: int
    documents_ingested: bool
    metadata_filter: Optional[dict]
    restrict_docs: Optional[list[str]]
    ingest_only: bool

    extracted_pages: list[dict]
    extraction_warnings: list[str]

    indexed_chunks_count: int
    index_status: str
    doc_types: dict

    answer_response: dict
    retrieved_chunks: list[dict]

    judge_verdict: dict

    human_score: Optional[dict]
    human_approved: Optional[bool]

    error: Optional[str]
    trace: Annotated[list[dict], operator.add]
