"""DocJudge — Streamlit UI.

Upload policy/finance PDFs, ask grounded questions, see the independent Judge
verdict, and approve/reject (human-in-the-loop). Answers are scoped to the
documents you uploaded this session, so a stale index never leaks in.

Run from repo root:
    streamlit run frontend/app.py
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
from langgraph.types import Command

from config import config
from graph import build_graph
from tools.file_extractors import SUPPORTED_EXTS
from observability.logging_config import setup_logging
from tools import chromadb_tool, ocr_tool

setup_logging()

st.set_page_config(page_title="DocJudge", page_icon=":material/balance:",
                   layout="wide", initial_sidebar_state="expanded")
