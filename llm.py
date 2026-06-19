"""Groq LLM client factory. Centralizes model selection + structured output."""
from __future__ import annotations

from functools import lru_cache
from typing import Type, TypeVar

from langchain_groq import ChatGroq
from pydantic import BaseModel

from config import config
