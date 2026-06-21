"""Agent 4: Answer Agent (Retriever + Generator).

Retrieves chunks from ChromaDB (metadata filter + HNSW similarity), then
generates a grounded answer that cites specific chunks. Strict refusal behavior.
"""
from __future__ import annotations

from config import config
from llm import invoke_structured
from models import AnswerResponse
from observability.logging_config import log_info, timed_node
from prompts.answer_prompt import ANSWER_SYSTEM, ANSWER_USER
from tools import chromadb_tool

def _docs_clause(restrict_docs: list[str]) -> dict:
    if len(restrict_docs) == 1:
        return {"doc_name": {"$eq": restrict_docs[0]}}
    return {"doc_name": {"$in": list(restrict_docs)}}

def _format_context(chunks: list[dict]) -> str:
    lines = []
    for c in chunks:
        lines.append(
            f"[{c['chunk_id']} | {c['doc_name']} | p{c['page']}]\n{c['text']}\n"
        )
    return "\n".join(lines) if lines else "(no chunks retrieved)"

def retrieve(question: str, metadata_filter: dict | None,
             restrict_docs: list[str] | None = None) -> list[dict]:
    """Retrieve top-k chunks.

    Multi-document rule: when the user has an explicit doc scope (the docs they
    uploaded this session), that scope WINS and we search across ALL of them. We
    do NOT also apply the orchestrator's inferred single-doc `doc_hint`, which
    would wrongly narrow a multi-PDF session to one file and miss cross-doc
    answers. The inferred filter is used only when there is no explicit scope.
    """
    if restrict_docs:
        scope = _docs_clause(restrict_docs)
        chunks = chromadb_tool.query(question, top_k=config.TOP_K, where=scope)
        if not chunks:
            chunks = chromadb_tool.query(question, top_k=config.TOP_K, where=None)
    else:
        chunks = chromadb_tool.query(question, top_k=config.TOP_K, where=metadata_filter)
        if not chunks and metadata_filter:
            chunks = chromadb_tool.query(question, top_k=config.TOP_K, where=None)
    log_info("answer", f"retrieved {len(chunks)} chunks "
                       f"(scope={'yes' if restrict_docs else 'no'}, docs={restrict_docs or 'all'})")
    return chunks

@timed_node("answer")
def answer_node(state: dict) -> dict:
    question = state["user_question"]
    metadata_filter = state.get("metadata_filter")
    retry_feedback = ""
    if state.get("retry_count", 0) > 0:
        verdict = state.get("judge_verdict") or {}
        human = state.get("human_score") or {}
        retry_feedback = verdict.get("feedback_for_retry") or human.get("feedback", "")

    chunks = retrieve(question, metadata_filter, state.get("restrict_docs"))

    if not chunks:
        resp = AnswerResponse(
            answer="The documents do not contain this information.",
            cited_chunks=[],
            confidence="not_found",
            refusal=True,
        )
        return {
            "answer_response": resp.model_dump(),
            "retrieved_chunks": chunks,
            "current_step": "answer:no_chunks",
        }

    context = _format_context(chunks)
    user_msg = ANSWER_USER.format(question=question, context=context)
    if retry_feedback:
        user_msg += f"\n\nA previous attempt failed verification. Fix this: {retry_feedback}"

    try:
        resp: AnswerResponse = invoke_structured(
            config.LLM_MODEL,
            AnswerResponse,
            [("system", ANSWER_SYSTEM), ("user", user_msg)],
        )
    except Exception as exc:
        resp = AnswerResponse(
            answer=f"Answer generation failed: {exc}",
            cited_chunks=[],
            confidence="not_found",
            refusal=True,
        )

    by_id = {c["chunk_id"]: c for c in chunks}
    for cited in resp.cited_chunks:
        src = by_id.get(cited.chunk_id)
        if src:
            cited.doc_name = src["doc_name"]
            cited.page = src["page"]

    return {
        "answer_response": resp.model_dump(),
        "retrieved_chunks": chunks,
        "current_step": f"answered (confidence={resp.confidence})",
    }
