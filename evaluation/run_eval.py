"""Run all evaluation cases unattended and print a pass/fail report.

HITL is auto-approved here so the suite runs headless. Run from repo root:
    python -m evaluation.run_eval
"""
from __future__ import annotations

import json
import os
import sys

from langgraph.types import Command

from config import config
from evaluation.test_cases import CASES
from graph import build_graph
from observability.logging_config import setup_logging
from tools import chromadb_tool

EVAL_CORPUS = [
    "data/sample_pdfs/annual_report_2025.pdf",
    "data/sample_pdfs/risk_policy.pdf",
]

def _ingest_corpus(app, thread) -> None:
    pdfs = [p for p in EVAL_CORPUS if os.path.exists(p)]
    if not pdfs:
        print("WARNING: no sample PDFs found. Run: python -m data.make_sample_pdfs")
    state = {
        "user_question": config.INGEST_SENTINEL,
        "documents": pdfs,
        "ingest_only": True,
        "retry_count": 0,
        "trace": [],
    }
    app.invoke(state, config=thread)

def _run_one(case, idx) -> tuple[bool, dict, dict]:
    app = build_graph()
    thread = {"configurable": {"thread_id": f"eval-{idx}"}}
    state = {"user_question": case.question, "documents": [], "retry_count": 0, "trace": []}
    res = app.invoke(state, config=thread)
    while "__interrupt__" in res:
        res = app.invoke(Command(resume={"decision": "approve"}), config=thread)
    ans = res.get("answer_response", {}) or {}
    verdict = res.get("judge_verdict", {}) or {}
    try:
        passed = bool(case.check(ans, verdict))
    except Exception as exc:
        passed = False
        ans["_check_error"] = str(exc)
    return passed, ans, verdict

def main() -> int:
    config.validate()
    setup_logging()

    chromadb_tool.reset()
    warm = build_graph()
    _ingest_corpus(warm, {"configurable": {"thread_id": "eval-ingest"}})

    print(f"\nIndexed chunks: {chromadb_tool.count()}\n")
    results = []
    for i, case in enumerate(CASES):
        passed, ans, verdict = _run_one(case, i)
        results.append((case, passed))
        mark = "PASS" if passed else "FAIL"
        print(f"[{mark}] #{case.id} {case.name} — {case.what_it_tests}")
        print(f"        Q: {case.question}")
        print(f"        refusal={ans.get('refusal')} confidence={ans.get('confidence')} "
              f"judge_pass={verdict.get('overall_pass')}")
        print()

    n_pass = sum(1 for _, p in results if p)
    print("=" * 60)
    print(f"RESULT: {n_pass}/{len(results)} cases passed")
    return 0 if n_pass == len(results) else 1

if __name__ == "__main__":
    sys.exit(main())
