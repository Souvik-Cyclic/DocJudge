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
