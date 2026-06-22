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
