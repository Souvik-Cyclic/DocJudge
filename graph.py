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
