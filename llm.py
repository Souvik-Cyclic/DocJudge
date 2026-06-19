"""Groq LLM client factory. Centralizes model selection + structured output."""
from __future__ import annotations

from functools import lru_cache
from typing import Type, TypeVar

from langchain_groq import ChatGroq
from pydantic import BaseModel

from config import config

T = TypeVar("T", bound=BaseModel)

@lru_cache(maxsize=4)
def get_llm(model: str | None = None, temperature: float | None = None) -> ChatGroq:
    """Return a cached ChatGroq client.

    Args:
        model: override model id; defaults to config.LLM_MODEL.
        temperature: override; defaults to config.LLM_TEMPERATURE.
    """
    config.validate()
    return ChatGroq(
        model=model or config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE if temperature is None else temperature,
        api_key=config.GROQ_API_KEY,
        max_retries=config.LLM_MAX_RETRIES,
    )

def get_judge_llm() -> ChatGroq:
    """Separate model for the Judge — different failure modes than the answer model."""
    return get_llm(model=config.JUDGE_MODEL)

def structured(model: str | None, schema: Type[T]):
    """Return an LLM that is forced to emit `schema` (Pydantic) via tool calling."""
    return get_llm(model=model).with_structured_output(schema)

def invoke_structured(
    model: str | None,
    schema: Type[T],
    messages: list,
    retries: int = 3,
) -> T:
    """Robust structured call for small/flaky models.

    Small models (e.g. llama-3.1-8b) intermittently emit malformed tool args
    (a field as a string instead of array/int). Groq rejects these server-side.
    We retry a few times (generation is stochastic) and, as a last resort, try
    to salvage the JSON embedded in the provider's `failed_generation` payload.

    Raises the last exception if every attempt fails.
    """
    import json
    import re

    runner = structured(model, schema)
    last_exc: Exception | None = None

    for _ in range(retries):
        try:
            return runner.invoke(messages)
        except Exception as exc:
            last_exc = exc
            salvaged = _salvage(exc, schema)
            if salvaged is not None:
                return salvaged

    raise last_exc if last_exc else RuntimeError("structured call failed")

def _salvage(exc: Exception, schema: Type[T]):
    """Best-effort: pull JSON from a failed_generation blob and validate it."""
    import json
    import re

    msg = str(exc)
    if "failed_generation" not in msg:
        return None
    m = re.search(r"failed_generation['\"]?:\s*['\"]?(.*)", msg, re.DOTALL)
    if not m:
        return None
    blob = m.group(1)
    obj = _first_json_object(blob)
    if obj is None:
        return None
    try:
        return schema.model_validate(obj)
    except Exception:
        return None

def _first_json_object(text: str):
    """Return the first brace-balanced JSON object parsed from `text`, or None."""
    import json

    start = text.find("{")
    if start == -1:
        return None
    depth, in_str, esc = 0, False, False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start : i + 1])
                except Exception:
                    return None
    return None
