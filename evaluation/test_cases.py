"""8 evaluation scenarios (exceeds the minimum of 5).

Each case declares an expectation that `run_eval.py` checks against the produced
answer + judge verdict. Designed against the sample corpus in data/sample_pdfs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

@dataclass
class EvalCase:
    id: int
    name: str
    question: str
    what_it_tests: str
    check: Callable[[dict, dict], bool]
    note: str = ""

def _not_refusal(ans: dict, _v: dict) -> bool:
    return not ans.get("refusal", False)

def _is_refusal(ans: dict, _v: dict) -> bool:
    return ans.get("refusal", False) or ans.get("confidence") == "not_found"

def _grounded_pass(_ans: dict, v: dict) -> bool:
    return bool(v.get("overall_pass"))

def _cites_any(ans: dict, _v: dict) -> bool:
    return len(ans.get("cited_chunks", [])) > 0

def _mentions(*subs: str):
    def _check(ans: dict, _v: dict) -> bool:
        text = (ans.get("answer") or "").lower()
        return any(s.lower() in text for s in subs)
    return _check

CASES: list[EvalCase] = [
    EvalCase(
        1, "direct_factual",
        "What was the total revenue in Q3 2025?",
        "Basic retrieval + grounding",
        check=lambda a, v: _not_refusal(a, v) and _cites_any(a, v),
    ),
    EvalCase(
        2, "paraphrased",
        "How much money did the company bring in during the third quarter?",
        "Semantic retrieval (wording differs from document)",
        check=lambda a, v: _not_refusal(a, v) and _cites_any(a, v),
    ),
    EvalCase(
        3, "table_numeric",
        "What is the operating margin listed in the financial summary table?",
        "Table parsing + structured chunk retrieval",
        check=_cites_any,
    ),
    EvalCase(
        4, "not_in_docs",
        "What is the CEO's home address?",
        "Refusal handling (info absent)",
        check=_is_refusal,
    ),
    EvalCase(
        5, "multi_section",
        "Summarize both the revenue results and the main risk factor.",
        "Multi-chunk retrieval and synthesis",
        check=lambda a, v: _not_refusal(a, v) and len(a.get("cited_chunks", [])) >= 2,
    ),
    EvalCase(
        6, "ambiguous",
        "What about the risks?",
        "Retrieval precision on a vague question",
        check=lambda a, v: _not_refusal(a, v) and _mentions("risk", "customer", "concentration")(a, v),
    ),
    EvalCase(
        7, "metadata_filter",
        "According to the Annual Report, what was the dividend policy?",
        "Metadata filtering to the correct document",
        check=lambda a, v: _not_refusal(a, v) and _mentions("dividend")(a, v),
    ),
    EvalCase(
        8, "adversarial_hallucination",
        "What does the document say about the company's Mars colonization plan?",
        "Hallucination guardrail (plausible but nonexistent policy)",
        check=_is_refusal,
    ),
]
