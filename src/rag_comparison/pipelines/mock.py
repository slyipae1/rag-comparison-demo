"""Deterministic offline pipeline for the quick demo.

No network calls, credentials, model downloads or external services are used.
It emits the same ``content`` / ``reasoning`` / ``inference_time`` shape as the
real pipelines so the API and UI can be exercised without lab infrastructure.
"""
from __future__ import annotations

import asyncio
import hashlib
import time
from typing import Any, Dict, AsyncIterator

from rag_comparison.pipelines.base import RAGpipeline


class MockPipeline(RAGpipeline):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.with_retrieval = config.get("RAG_pipeline_type") == "mock_rag"
        self.delay = float(config.get("mock_delay_seconds", 0.02))
        self.record_timing = bool(config.get("record_timing", False))

    def _context(self, question: str) -> str:
        digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:8]
        return (
            f"Synthetic document #{digest}: this repository demonstrates a "
            "configuration-driven comparison framework, not a benchmark."
        )

    async def process_question(self, question: str, use_stream: bool = False) -> AsyncIterator[Dict[str, Any]]:
        started = time.time()
        reasoning = ""
        if self.with_retrieval:
            await asyncio.sleep(self.delay)
            reasoning = f"## Synthetic retrieval\n{self._context(question)}\n"
        else:
            await asyncio.sleep(max(self.delay / 4, 0.005))
            reasoning = "No retrieval stage (LLM-only configuration)."

        answer = (
            f"[mock answer] You asked: {question}\n"
            + ("Retrieval context was included." if self.with_retrieval else "No context was used.")
        )

        if use_stream:
            yield {"reasoning": reasoning}
            total = time.time() - started
            yield {"inference_time": total}
            words = answer.split(" ")
            for word in words:
                await asyncio.sleep(0.001)
                yield {"content": word + " "}
        else:
            yield {
                "content": answer,
                "reasoning": reasoning,
                "inference_time": time.time() - started,
            }

        if self.record_timing:
            from rag_comparison.instrumentation.timing import RecordTime
            RecordTime({
                "PIPELINE": self.config.get("RAG_pipeline_type"),
                "TOTAL": time.time() - started,
                "QUESTION": question,
            })
