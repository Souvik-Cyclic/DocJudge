"""LangGraph StateGraph wiring DocJudge's 5 agents + HITL.

Flow:
  orchestrator -> (ingest? ) -> extraction -> structurer -> answer -> judge
                  (already ingested) ------------------------> answer -> judge
  judge -> (pass) -> human_review (HITL interrupt) -> END
  judge -> (fail & retries left) -> answer  (retry with feedback)
  judge -> (fail & out of retries) -> human_review (surface best attempt)

3 conditional routing points:
  1. orchestrator -> extraction vs answer   (documents_ingested)
  2. judge -> answer (retry) vs human_review (overall_pass / retry_count)
  3. judge refusal re-check folded into the judge verdict (refusal_appropriate)
"""
from __future__ import annotations

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from agents.answer_agent import answer_node
from agents.extraction_agent import extraction_node
from agents.judge_agent import judge_node
from agents.orchestrator import orchestrator_node
from agents.structurer_agent import structurer_node
from config import config
from observability.logging_config import timed_node
from state import GraphState

@timed_node("human_review")
def human_review_node(state: dict) -> dict:
    """Pause for a human to score + approve/reject (high-impact gate).

    `interrupt()` suspends the graph; the caller resumes with a Command payload:
        {"relevance": int, "grounding": int, "citation": int,
         "decision": "approve"|"reject", "feedback": str}
    """
    payload = {
        "answer": state.get("answer_response", {}),
        "judge_verdict": state.get("judge_verdict", {}),
        "retrieved_chunks": state.get("retrieved_chunks", []),
    }
    human = interrupt(payload)

    decision = (human or {}).get("decision", "approve")
    return {
        "human_score": human,
        "human_approved": decision == "approve",
        "current_step": f"human:{decision}",
    }

def route_after_orchestrator(state: dict) -> str:
    if state.get("current_step") == "rejected":
        return "human_review"
    if state.get("ingest_only"):
        return "extraction"
    return "answer" if state.get("documents_ingested") else "extraction"
