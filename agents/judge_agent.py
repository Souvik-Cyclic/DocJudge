"""Agent 5: Judge Agent (Verifier).

Independently evaluates the answer for grounding, hallucination, citation
accuracy, and refusal correctness. It judges only; it never answers. Runs on a
DIFFERENT model than the Answer Agent to reduce shared blind spots.

Includes a DETERMINISTIC citation pre-check (string matching) alongside the
LLM-based entailment check — so the guardrail does not rely on the LLM alone.
"""
from __future__ import annotations

from config import config
from llm import invoke_structured
from models import JudgeVerdict
from observability.logging_config import timed_node
from prompts.judge_prompt import JUDGE_SYSTEM, JUDGE_USER

def _deterministic_citation_check(answer_response: dict, retrieved: list[dict]) -> tuple[bool, list[str]]:
    """Verify each cited chunk_id was actually retrieved and snippet overlaps text.

    Returns (citations_valid, issues).
    """
    issues: list[str] = []
    retrieved_by_id = {c["chunk_id"]: c["text"] for c in retrieved}
    cited = answer_response.get("cited_chunks", [])

    if answer_response.get("refusal"):
        return True, issues

    if not cited:
        return False, ["Answer makes claims but cites no chunks."]

    valid = True
    for c in cited:
        cid = c.get("chunk_id", "")
        if cid not in retrieved_by_id:
            valid = False
            issues.append(f"Cited chunk_id '{cid}' was not in the retrieved set.")
            continue
        snippet = (c.get("snippet") or "").strip().lower()
        source = retrieved_by_id[cid].lower()
        if snippet and len(snippet) >= 12 and snippet[:40] not in source and snippet not in source:
            tokens = [t for t in snippet.split() if len(t) > 3]
            hits = sum(1 for t in tokens if t in source)
            if not tokens or hits / len(tokens) < 0.5:
                valid = False
                issues.append(f"Snippet for '{cid}' not found in the cited chunk text.")
    return valid, issues

def _format_context(chunks: list[dict]) -> str:
    return "\n".join(
        f"[{c['chunk_id']} | {c['doc_name']} | p{c['page']}]\n{c['text']}\n" for c in chunks
    ) or "(no chunks)"

@timed_node("judge")
def judge_node(state: dict) -> dict:
    answer_response = state.get("answer_response", {}) or {}
    retrieved = state.get("retrieved_chunks", []) or []
    question = state["user_question"]
    is_refusal = bool(answer_response.get("refusal"))

    det_valid, det_issues = _deterministic_citation_check(answer_response, retrieved)

    citations_str = "\n".join(
        f"- {c.get('chunk_id')}: {c.get('snippet')}"
        for c in answer_response.get("cited_chunks", [])
    ) or "(none)"

    user_msg = JUDGE_USER.format(
        question=question,
        context=_format_context(retrieved),
        answer=answer_response.get("answer", ""),
        citations=citations_str,
        is_refusal="yes" if is_refusal else "no",
    )

    try:
        verdict: JudgeVerdict = invoke_structured(
            config.JUDGE_MODEL,
            JudgeVerdict,
            [("system", JUDGE_SYSTEM), ("user", user_msg)],
        )
    except Exception as exc:
        verdict = JudgeVerdict(
            grounded=False,
            hallucination_detected=True,
            citations_valid=det_valid,
            overall_pass=False,
            issues=[f"Judge LLM error: {exc}"],
            feedback_for_retry="Judge failed; regenerate a strictly grounded answer.",
        )

    if not det_valid:
        verdict.citations_valid = False
        verdict.overall_pass = False
        verdict.issues = list(set(verdict.issues + det_issues))
        if not verdict.feedback_for_retry:
            verdict.feedback_for_retry = "Fix invalid citations: " + "; ".join(det_issues)

    return {
        "judge_verdict": verdict.model_dump(),
        "current_step": f"judged (pass={verdict.overall_pass})",
    }
