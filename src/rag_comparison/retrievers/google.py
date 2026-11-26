"""Optional web search retriever backed by SerpAPI.

The API key is read from the environment (``SERPAPI_API_KEY``) or from the
pipeline config.  An optional local cache file can be supplied for offline
replays; no real search results are bundled with this repository.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict

from langchain_community.utilities import SerpAPIWrapper
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from rag_comparison.retrievers.base import Retriever

logger = logging.getLogger(__name__)


class GoogleRetriever(Retriever):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.cache_path = config.get("search_cache_path") or os.getenv("GOOGLE_CACHE_PATH")
        self.summarization_prompt = ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a concise summarizer. Preserve the original language and all "
                "informative content, remove formatting noise, and extract key facts, "
                "especially numbers and times. Answer in the same language as the question.",
            ),
            (
                "user",
                "Summarize the following material, keeping key facts, numbers and times. "
                "Answer in the same language as the question: {text}",
            ),
        ])

    def batch(self, queries):
        return [self.invoke(q) for q in queries]

    def _load_cached(self) -> Dict[str, str]:
        if not self.cache_path or not os.path.exists(self.cache_path):
            return {}
        results: Dict[str, str] = {}
        with open(self.cache_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for i in range(0, len(lines), 2):
            if i + 1 >= len(lines):
                break
            q = lines[i].strip().removeprefix("query: ")
            text = lines[i + 1].strip().removeprefix("raw_context: ")
            results[q.strip()] = text
        return results

    def invoke(self, query: str) -> str:
        cached = self._load_cached()
        if query.strip() in cached:
            logger.info("Using cached search results for query")
            return cached[query.strip()]

        api_key = self.config.get("serpapi_api_key") or os.getenv("SERPAPI_API_KEY")
        if not api_key:
            logger.warning("SERPAPI_API_KEY is not set; returning empty search context")
            return ""

        search = SerpAPIWrapper(params={"engine": "google", "gl": "hk"}, serpapi_api_key=api_key)
        try:
            results = search.results(query)
        except Exception as exc:
            logger.error("Google search failed: %s", exc)
            return ""
        context = ""
        if results:
            context = "\n".join(f"{key}: {value}" for key, value in results.items())
        if self.cache_path:
            with open(self.cache_path, "a", encoding="utf-8") as f:
                f.write(f"query: {query}\nraw_context: {context}\n")
        return context

    def summarize_google_results(self, raw_context, internal_llm):
        if not raw_context:
            return ""
        summary = (self.summarization_prompt | internal_llm | StrOutputParser()).invoke(
            {"text": raw_context}
        )
        return summary.strip()
