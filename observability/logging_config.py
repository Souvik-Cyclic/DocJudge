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
