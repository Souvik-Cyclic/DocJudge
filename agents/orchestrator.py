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
