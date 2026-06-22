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
