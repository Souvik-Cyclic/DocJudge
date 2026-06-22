"""CLI entrypoint for DocJudge.

Usage:
  python main.py --ingest data/sample_pdfs/*.pdf --question "What was Q3 revenue?"
  python main.py --question "What was Q3 revenue?"          # use existing index

The graph pauses at the human-review node (HITL). In the CLI we collect scores
interactively and resume.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

from langgraph.types import Command

from config import config
from graph import build_graph
from observability.logging_config import setup_logging

def _expand(patterns: list[str]) -> list[str]:
    files: list[str] = []
    for p in patterns:
        files.extend(glob.glob(p))
    return sorted(set(files))

def _collect_human_input(payload: dict) -> dict:
    ans = payload.get("answer", {})
    verdict = payload.get("judge_verdict", {})
    print("\n" + "=" * 70)
    print("HUMAN REVIEW (high-impact gate)")
    print("=" * 70)
    print(f"\nAnswer:\n{ans.get('answer')}")
    print(f"\nConfidence: {ans.get('confidence')}  Refusal: {ans.get('refusal')}")
    print("\nCited chunks:")
    for c in ans.get("cited_chunks", []):
        print(f"  - {c.get('chunk_id')} ({c.get('doc_name')} p{c.get('page')})")
    print(f"\nJudge verdict: pass={verdict.get('overall_pass')} "
          f"grounded={verdict.get('grounded')} "
          f"hallucination={verdict.get('hallucination_detected')}")
    if verdict.get("issues"):
        print("Judge issues: " + "; ".join(verdict["issues"]))

    def _score(label: str) -> int:
        while True:
            raw = input(f"  {label} score (1-5): ").strip()
            if raw.isdigit() and 1 <= int(raw) <= 5:
                return int(raw)

    relevance = _score("Relevance")
    grounding = _score("Grounding")
    citation = _score("Citation quality")
    decision = ""
    while decision not in {"approve", "reject"}:
        decision = input("  Decision [approve/reject]: ").strip().lower()
    feedback = ""
    if decision == "reject":
        feedback = input("  Feedback for retry: ").strip()
    return {
        "relevance": relevance,
        "grounding": grounding,
        "citation": citation,
        "decision": decision,
        "feedback": feedback,
    }

def run(question: str, documents: list[str]) -> dict:
    config.validate()
    setup_logging()
    app = build_graph()
    thread = {"configurable": {"thread_id": "cli-session"}}

    restrict = [os.path.basename(p) for p in documents] if documents else None

    state = {
        "user_question": question,
        "documents": documents,
        "restrict_docs": restrict,
        "retry_count": 0,
        "trace": [],
    }

    result = app.invoke(state, config=thread)
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        human = _collect_human_input(payload)
        result = app.invoke(Command(resume=human), config=thread)

    return result

def main() -> None:
    ap = argparse.ArgumentParser(description="DocJudge — multi-agent document Q&A")
    ap.add_argument("--question", "-q", required=True, help="user question")
    ap.add_argument("--ingest", "-i", nargs="*", default=[], help="PDF paths/globs")
    args = ap.parse_args()

    docs = _expand(args.ingest)
    final = run(args.question, docs)

    print("\n" + "=" * 70)
    print("FINAL STATE")
    print("=" * 70)
    print(json.dumps(
        {
            "answer": final.get("answer_response"),
            "judge_verdict": final.get("judge_verdict"),
            "human_approved": final.get("human_approved"),
            "indexed_chunks_count": final.get("indexed_chunks_count"),
        },
        indent=2,
        default=str,
    ))

if __name__ == "__main__":
    main()
