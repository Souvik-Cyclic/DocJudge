"""Central configuration. All env-driven so no secrets in code."""
from __future__ import annotations

import os

os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

from dotenv import load_dotenv

load_dotenv()

_tracing_on = (os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true"
               and bool(os.getenv("LANGCHAIN_API_KEY")))
if _tracing_on:
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = os.environ["LANGCHAIN_API_KEY"]
    if os.getenv("LANGCHAIN_PROJECT"):
        os.environ["LANGSMITH_PROJECT"] = os.environ["LANGCHAIN_PROJECT"]
else:
    os.environ["LANGCHAIN_TRACING_V2"] = "false"
    os.environ["LANGSMITH_TRACING"] = "false"

class Config:
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    LLM_MODEL: str = os.getenv("DOCJUDGE_LLM_MODEL", "llama-3.3-70b-versatile")
    JUDGE_MODEL: str = os.getenv("DOCJUDGE_JUDGE_MODEL", "qwen/qwen3-32b")
    LLM_TEMPERATURE: float = 0.0
    LLM_MAX_RETRIES: int = int(os.getenv("DOCJUDGE_LLM_MAX_RETRIES", "5"))

    EMBED_MODEL: str = os.getenv("DOCJUDGE_EMBED_MODEL", "BAAI/bge-small-en-v1.5")

    CHROMA_DIR: str = os.getenv("DOCJUDGE_CHROMA_DIR", "./data/chroma")
    CHROMA_COLLECTION: str = os.getenv("DOCJUDGE_CHROMA_COLLECTION", "docjudge")

    TOP_K: int = int(os.getenv("DOCJUDGE_TOP_K", "10"))
    MAX_RETRIES: int = int(os.getenv("DOCJUDGE_MAX_RETRIES", "2"))

    MAX_QUESTION_LEN: int = 2000
    MIN_QUESTION_LEN: int = 3

    LOG_DIR: str = os.getenv("DOCJUDGE_LOG_DIR", "./logs")
    LOG_FILE: str = os.getenv("DOCJUDGE_LOG_FILE", "docjudge.log")
    LOG_LEVEL: str = os.getenv("DOCJUDGE_LOG_LEVEL", "INFO")

    INGEST_SENTINEL: str = "__ingest_only__"

    LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "false")
    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT: str = os.getenv("LANGCHAIN_PROJECT", "docjudge")

    @property
    def tracing_enabled(self) -> bool:
        return self.LANGCHAIN_TRACING_V2.lower() == "true" and bool(self.LANGCHAIN_API_KEY)

    @classmethod
    def validate(cls) -> None:
        if not cls.GROQ_API_KEY:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
            )

config = Config()
