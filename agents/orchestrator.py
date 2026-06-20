"""Agent 1: Orchestrator (Planner).

Real work, not just routing:
  - validates input (guardrail)
  - decides if ingestion is needed
  - analyzes the question to infer a ChromaDB metadata filter + retrieval hints
  - owns retry bookkeeping
"""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from config import config
from llm import invoke_structured
from observability.logging_config import timed_node
from prompts.orchestrator_prompt import ORCHESTRATOR_SYSTEM, ORCHESTRATOR_USER
from tools import chromadb_tool

class QuestionPlan(BaseModel):
    doc_hint: Optional[str] = None
    section_hint: Optional[str] = None
    type_hint: Optional[str] = None
    needs_table: bool = False
    reasoning: str = ""

def _validate_question(q: str) -> Optional[str]:
    """Input guardrail. Returns an error message or None."""
    if not q or not q.strip():
        return "Empty question."
    if len(q) < config.MIN_QUESTION_LEN:
        return "Question too short."
    if len(q) > config.MAX_QUESTION_LEN:
        return f"Question exceeds {config.MAX_QUESTION_LEN} chars."
    return None

def _infer_filter(plan: QuestionPlan) -> Optional[dict]:
    """Turn hints into a ChromaDB `where` filter (or None)."""
    clauses = []
    if plan.doc_hint:
        clauses.append({"doc_name": {"$eq": plan.doc_hint}})
    if plan.section_hint:
        clauses.append({"section": {"$eq": plan.section_hint}})
    if plan.type_hint in ("policy", "financial"):
        clauses.append({"doc_type": {"$eq": plan.type_hint}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}

@timed_node("orchestrator")
def orchestrator_node(state: dict) -> dict:
    question = state.get("user_question", "")
    docs_in = state.get("documents", []) or []

    if question == config.INGEST_SENTINEL or (docs_in and not question.strip()):
        return {
            "ingest_only": True,
            "documents_ingested": False,
            "execution_plan": "ingest",
            "current_step": "ingest_only",
            "retry_count": 0,
        }

    err = _validate_question(question)
    if err:
        return {
            "error": err,
            "current_step": "rejected",
            "answer_response": {
                "answer": f"Request rejected by guardrail: {err}",
                "cited_chunks": [],
                "confidence": "not_found",
                "refusal": True,
            },
        }

    docs = state.get("documents", []) or []
    already = chromadb_tool.count() > 0
    documents_ingested = already and not docs

    metadata_filter = state.get("metadata_filter")
    if metadata_filter is None and not state.get("retry_count"):
        known = chromadb_tool.list_doc_names() or docs
        try:
            plan: QuestionPlan = invoke_structured(
                config.LLM_MODEL,
                QuestionPlan,
                [
                    ("system", ORCHESTRATOR_SYSTEM),
                    (
                        "user",
                        ORCHESTRATOR_USER.format(
                            question=question,
                            doc_names=", ".join(known) or "unknown",
                        ),
                    ),
                ],
            )
            if plan.doc_hint and known and plan.doc_hint not in known:
                plan.doc_hint = None
            metadata_filter = _infer_filter(plan)
        except Exception:
            metadata_filter = None

    step = "answer" if documents_ingested else "ingest"
    return {
        "documents_ingested": documents_ingested,
        "metadata_filter": metadata_filter,
        "execution_plan": step,
        "current_step": step,
        "retry_count": state.get("retry_count", 0),
        "ingest_only": False,
    }
