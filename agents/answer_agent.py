"""Agent 4: Answer Agent (Retriever + Generator).

Retrieves chunks from ChromaDB (metadata filter + HNSW similarity), then
generates a grounded answer that cites specific chunks. Strict refusal behavior.
"""
from __future__ import annotations

from config import config
from llm import invoke_structured
