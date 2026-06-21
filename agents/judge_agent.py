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
