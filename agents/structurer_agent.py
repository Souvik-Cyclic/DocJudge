"""Agent 3: Structurer Agent (type-aware chunker + indexer).

Per document:
  1. classify it (policy / financial / general)  -> tools.classifier
  2. route to the chunking strategy tuned for that type -> tools.chunkers
  3. label chunks (section / content_type / topic_tags) and generate a
     "context_prefix" that situates each chunk (contextual retrieval)
  4. embed (BGE-small) + store in ChromaDB with doc_type metadata.

The type-aware routing means a legal clause, a financial table, and ordinary
prose are no longer cut by the same fixed-size window.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from config import config
from llm import invoke_structured
from models import IndexedChunk
from observability.logging_config import log_info, timed_node
from prompts.structurer_prompt import STRUCTURER_SYSTEM
from tools import chromadb_tool
