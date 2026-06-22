"""Observability: human-readable console + persistent file logging, plus a
node-timing decorator that records a structured trace in graph state.

Two log streams:
  - console: clean, level-colored-ish prefixed lines for the demo
  - file (logs/docjudge.log): full detail, one line per event, kept across runs

LangSmith tracing (if LANGCHAIN_TRACING_V2=true) runs on top of this; this
module is the always-on fallback so debugging never depends on LangSmith.
"""
from __future__ import annotations

import functools
import json
import logging
import os
import sys
import time
from logging.handlers import RotatingFileHandler
from typing import Callable

from config import config

logger = logging.getLogger("docjudge")

try:
    from langgraph.errors import GraphBubbleUp as _CONTROL_FLOW_EXC
except Exception:
    try:
        from langgraph.errors import GraphInterrupt as _CONTROL_FLOW_EXC
    except Exception:
        _CONTROL_FLOW_EXC = ()

def setup_logging(level: str | int | None = None) -> None:
    """Idempotently configure console + rotating file handlers."""
    if logger.handlers:
        return
    lvl = level or config.LOG_LEVEL
    if isinstance(lvl, str):
        lvl = getattr(logging, lvl.upper(), logging.INFO)
    logger.setLevel(lvl)
    logger.propagate = False

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-5s | %(message)s",
                                      datefmt="%H:%M:%S"))
    logger.addHandler(ch)

    try:
        os.makedirs(config.LOG_DIR, exist_ok=True)
        fh = RotatingFileHandler(
            os.path.join(config.LOG_DIR, config.LOG_FILE),
            maxBytes=2_000_000, backupCount=3, encoding="utf-8",
        )
        fh.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)-5s | %(message)s"))
        logger.addHandler(fh)
    except Exception as exc:
        logger.warning("file logging disabled: %s", exc)

    _log_tracing_status()

def _log_tracing_status() -> None:
    if config.tracing_enabled:
        logger.info(f"LangSmith tracing ENABLED -> project '{config.LANGCHAIN_PROJECT}' "
                    f"(view runs at https://smith.langchain.com)")
    elif config.LANGCHAIN_TRACING_V2.lower() == "true":
        logger.warning("LANGCHAIN_TRACING_V2=true but LANGCHAIN_API_KEY is missing "
                       "-> LangSmith tracing OFF (using file/console logs only)")
    else:
        logger.info("LangSmith tracing disabled (file/console logs only). "
                    "Set LANGCHAIN_TRACING_V2=true + LANGCHAIN_API_KEY in .env to enable.")

def log_event(node: str, level: int = logging.INFO, **fields) -> dict:
    """Emit one structured event (also returned for inclusion in state.trace)."""
    record = {"node": node, **fields}
    extras = " ".join(f"{k}={_short(v)}" for k, v in fields.items())
    logger.log(level, f"[{node}] {extras}")
    return record

def log_info(node: str, msg: str) -> None:
    logger.info(f"[{node}] {msg}")

def _short(v, n: int = 120) -> str:
    s = json.dumps(v, default=str) if not isinstance(v, str) else v
    return s if len(s) <= n else s[: n - 1] + "…"

def timed_node(node_name: str) -> Callable:
    """Decorator: time a node, log start/end, append a trace entry, catch errors.

    Wrapped fn: fn(state) -> dict (partial state update).
    Control-flow exceptions (HITL interrupt) are re-raised, never swallowed.
    On a real error, writes `error` to state instead of crashing the graph.
    """

    def decorator(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(state: dict) -> dict:
            logger.info(f"[{node_name}] START")
            start = time.perf_counter()
            try:
                update = fn(state) or {}
                latency = round(time.perf_counter() - start, 3)
                entry = log_event(
                    node_name,
                    latency_s=latency,
                    step=update.get("current_step", "ok"),
                )
            except _CONTROL_FLOW_EXC:
                logger.info(f"[{node_name}] PAUSE (awaiting human review)")
                raise
            except Exception as exc:
                latency = round(time.perf_counter() - start, 3)
                logger.error(f"[{node_name}] ERROR after {latency}s: {exc}")
                entry = {"node": node_name, "latency_s": latency, "error": str(exc)}
                return {"error": f"{node_name}: {exc}", "trace": [entry]}
            logger.info(f"[{node_name}] DONE in {latency}s")
            update["trace"] = [entry]
            return update

        return wrapper

    return decorator
