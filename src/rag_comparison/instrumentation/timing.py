"""Append stage timing records to a local JSONL file."""
from __future__ import annotations

import os
from datetime import datetime

from rag_comparison.config import settings


def RecordTime(time_dict):
    """Append one timing record.

    Expected keys: ``PIPELINE``, ``TOTAL``, ``QUESTION`` and optional
    stage-specific keys such as ``LLM_ANSWER`` / ``DB_Retrieval``.
    """
    os.makedirs(settings.LOG_DIR, exist_ok=True)
    record_path = os.path.join(settings.LOG_DIR, "time_record.jsonl")
    record = dict(time_dict)
    record["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(record_path, "a", encoding="utf-8") as f:
        f.write(f"{record}\n")
