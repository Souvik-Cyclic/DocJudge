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

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
.dj-hero {background: linear-gradient(135deg,#1e3a5f 0%,#2d6a9f 100%);
          padding: 1.3rem 1.6rem; border-radius: 12px; color: #fff; margin-bottom: 1.1rem;}
.dj-hero h1 {margin:0; font-size:1.55rem; letter-spacing:.3px;}
.dj-hero p {margin:.3rem 0 0; opacity:.9; font-size:.92rem;}
.dj-pill {display:inline-block; padding:.12rem .6rem; border-radius:6px;
          font-size:.72rem; font-weight:600; letter-spacing:.4px;}
.dj-step {padding:.4rem .65rem; border-radius:6px; background:#1e293b;
          color:#e2e8f0 !important; margin:.25rem 0; font-size:.82rem;
          border-left:3px solid #2d6a9f;}
.dj-step b {color:#fff;}
.dj-step span {color:#94a3b8 !important;}
.dj-pipe {padding:.8rem 1rem; border-radius:10px; background:#0f172a;
          color:#e2e8f0; border:1px solid #1e293b; line-height:1.9;
          font-size:.92rem; margin:.4rem 0; font-family:ui-monospace,monospace;}
.dj-pipe b {color:#7dd3fc; font-family:system-ui;}
.dj-doc {display:inline-block; padding:.12rem .55rem; border-radius:6px;
         background:#1e293b; color:#cbd5e1; font-size:.78rem; margin:.15rem .15rem 0 0;}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="dj-hero">
  <h1>DocJudge</h1>
  <p>Multi-agent policy &amp; finance document reviewer — grounded answers,
     independently verified, human-approved.</p>
</div>
""", unsafe_allow_html=True)

if "app" not in st.session_state:
    st.session_state.app = build_graph()
if "thread" not in st.session_state:
    st.session_state.thread = {"configurable": {"thread_id": "ui-session"}}
if "session_docs" not in st.session_state:
    st.session_state.session_docs = []
if "history" not in st.session_state:
    st.session_state.history = []
if "pending" not in st.session_state:
    st.session_state.pending = None
if "pending_q" not in st.session_state:
    st.session_state.pending_q = ""

app = st.session_state.app
thread = st.session_state.thread

AGENTS = [
    ("1", "Orchestrator", "validate, plan, route"),
    ("2", "Extraction", "PyMuPDF, pdfplumber, OCR"),
    ("3", "Structurer", "chunk, embed, index"),
    ("4", "Answer", "retrieve, grounded generation"),
    ("5", "Judge", "verify grounding / hallucination"),
    ("6", "Human review", "score, approve / reject"),
]
PIPE = [
    ("orchestrator", "1. Orchestrator"),
    ("extraction", "2. Extraction"),
    ("structurer", "3. Structurer"),
    ("answer", "4. Answer"),
    ("judge", "5. Judge"),
    ("human_review", "6. Human review"),
]

def run_stream(payload, thread):
    """Run the graph, lighting up each agent as it completes.

    Returns the final merged state, or {"__interrupt__": ...} if it paused for
    human review.
    """
    placeholder = st.empty()
    order = [k for k, _ in PIPE]
    done: set[str] = set()
    interrupt = None

    def render(finished: bool = False):
        done_idxs = [i for i, k in enumerate(order) if k in done]
        max_done = max(done_idxs) if done_idxs else -1
        running = None
        if not finished:
            running = next((k for i, k in enumerate(order)
                            if k not in done and i > max_done), None)
            if running is None and not done:
                running = order[0]
        rows = []
        for i, (key, label) in enumerate(PIPE):
            if key in done:
                mark = "[x]"
            elif key == running:
                mark = "[>]"
            elif i < max_done:
                mark = "[-]"
            else:
                mark = "[ ]"
            rows.append(f"{mark}&nbsp; {label}")
        placeholder.markdown(
            "<div class='dj-pipe'><b>PIPELINE</b><br>" + "<br>".join(rows) + "</div>",
            unsafe_allow_html=True,
        )

    render()
    for chunk in app.stream(payload, config=thread, stream_mode="updates"):
        for node in chunk:
            if node == "__interrupt__":
                interrupt = chunk["__interrupt__"]
                continue
            if node in order:
                done.add(node)
        render()

    render(finished=True)
    placeholder.empty()
    if interrupt is not None:
        return {"__interrupt__": interrupt}
    return app.get_state(thread).values

with st.sidebar:
    st.subheader("Documents")
    c1, c2 = st.columns(2)
    c1.metric("Indexed chunks", chromadb_tool.count())
    c2.metric("This session", len(st.session_state.session_docs))

    uploads = st.file_uploader(
        "Upload documents",
        type=["pdf", "docx", "pptx", "xlsx", "xls", "csv", "txt", "md",
              "png", "jpg", "jpeg", "tiff", "bmp", "webp"],
        accept_multiple_files=True)
    st.caption("PDF, Word, PowerPoint, Excel/CSV, text, images (OCR). No size/page limit.")
    if st.button("Ingest", use_container_width=True, type="primary") and uploads:
        os.makedirs("data/uploads", exist_ok=True)
        paths = []
        for up in uploads:
            fp = os.path.join("data/uploads", up.name)
            with open(fp, "wb") as f:
                f.write(up.getbuffer())
            paths.append(fp)
            if up.name not in st.session_state.session_docs:
                st.session_state.session_docs.append(up.name)
        with st.spinner("Extract, structure, embed, index..."):
            result = app.invoke(
                {"user_question": config.INGEST_SENTINEL, "documents": paths,
                 "ingest_only": True, "retry_count": 0, "trace": []},
                config=thread,
            )
        st.success(f"Indexed {len(paths)} file(s). Total chunks: {chromadb_tool.count()}")
        for warn in (result.get("extraction_warnings") or []):
            st.warning(warn)

    if st.session_state.session_docs:
        st.caption("Answering from:")
        st.markdown("".join(
            f"<span class='dj-doc'>{d}</span>" for d in st.session_state.session_docs),
            unsafe_allow_html=True)

    scope_only = st.toggle("Scope answers to uploaded docs", value=True,
                           help="Off = search the entire index (all docs ever ingested)")
    show_inspector = st.toggle("Show index inspector", value=False,
                               help="View the extracted/OCR text and how each "
                                    "document was classified and chunked")

    _ocr = ocr_tool.ocr_status()
    st.caption("OCR: ✅ available" if _ocr["available"]
               else f"OCR: ⚠️ unavailable — {_ocr['reason']}")
    st.caption(f"LangSmith: ✅ on (project={config.LANGCHAIN_PROJECT})"
               if config.tracing_enabled
               else "LangSmith: off (file/console logging active)")

    st.divider()
    if st.button("Reset index", use_container_width=True):
        chromadb_tool.reset()
        st.session_state.app = build_graph()
        st.session_state.session_docs = []
        st.session_state.history = []
        st.session_state.pending = None
        st.rerun()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.history = []
        st.session_state.pending = None
        st.rerun()

    st.divider()
    st.caption("Pipeline")
    for num, name, desc in AGENTS:
        st.markdown(f"<div class='dj-step'>{num}. <b>{name}</b><br>"
                    f"<span>{desc}</span></div>", unsafe_allow_html=True)

_CONF_COLORS = {"high": "#16a34a", "medium": "#d97706", "low": "#dc2626",
                "not_found": "#64748b", "?": "#64748b"}
