"""FastAPI endpoint exposing each configured pipeline as an OpenAI-style route.

Quick start (offline)::

    python -m rag_comparison.api.app --config-file configs/mock-demo.json --all

The server binds to 127.0.0.1 by default. Use ``--host`` explicitly for remote
deployment.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage]
    stream: bool = False


def build_pipeline(config: Dict[str, Any]):
    """Instantiate the pipeline named by ``RAG_pipeline_type``."""
    pipeline_type = config["RAG_pipeline_type"]
    if pipeline_type in {"mock_llm_only", "mock_rag"}:
        from rag_comparison.pipelines.mock import MockPipeline
        return MockPipeline(config)
    if pipeline_type == "LLM_Google":
        from rag_comparison.pipelines.llm_google import LLM_Google
        return LLM_Google(config)
    if pipeline_type == "LLM_Google_DB":
        from rag_comparison.pipelines.llm_google_db import LLM_Google_DB
        return LLM_Google_DB(config)
    if pipeline_type == "LLM_Google_Graph_RAG":
        from rag_comparison.pipelines.llm_google_graph_rag import LLM_Google_Graph_RAG
        return LLM_Google_Graph_RAG(config)
    raise ValueError(f"Unknown RAG_pipeline_type: {pipeline_type}")


def create_app(name: str, config: Dict[str, Any]) -> FastAPI:
    app = FastAPI(title=f"RAG Comparison: {name}")
    allow_origins = config.get("cors_origins", ["http://127.0.0.1:7860"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    pipeline = None

    def get_pipeline():
        nonlocal pipeline
        if pipeline is None:
            logger.info("Initializing pipeline %s", config.get("RAG_pipeline_type"))
            pipeline = build_pipeline(config)
        return pipeline

    @app.on_event("startup")
    async def startup_event():
        get_pipeline()

    @app.get("/health")
    def health_check():
        if pipeline is None:
            raise HTTPException(status_code=503, detail="Pipeline not initialized")
        return {"status": "healthy", "pipeline": config.get("RAG_pipeline_type")}

    @app.post("/v1/chat/completions")
    async def chat_completions(request: ChatRequest):
        question = next(
            (m.content for m in reversed(request.messages) if m.role == "user"), None
        )
        if not question:
            raise HTTPException(status_code=400, detail="No user message found")
        instance = get_pipeline()

        if request.stream:
            async def generate():
                async for chunk in instance.process_question(question, use_stream=True):
                    content = chunk.get("content", "")
                    reasoning = chunk.get("reasoning", "")
                    inference_time = chunk.get("inference_time")
                    if not (content or reasoning or inference_time):
                        continue
                    data = {
                        "choices": [{
                            "delta": {"role": "assistant", "content": content},
                            "inference_time": inference_time,
                            "reasoning_process": reasoning,
                        }]
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                yield "data: [DONE]\n\n"
            return StreamingResponse(generate(), media_type="text/event-stream")

        content_parts, reasoning_parts, inference_time = [], [], None
        async for chunk in instance.process_question(question, use_stream=False):
            if "content" in chunk:
                content_parts.append(chunk["content"])
            if "reasoning" in chunk:
                reasoning_parts.append(chunk["reasoning"])
            if "inference_time" in chunk:
                inference_time = chunk["inference_time"]
        return {
            "choices": [{
                "message": {"role": "assistant", "content": "".join(content_parts)},
                "inference_time": inference_time,
                "reasoning_process": "\n".join(reasoning_parts),
            }]
        }

    return app


def load_configs(path: str | Path) -> Dict[str, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAG comparison API endpoint(s)")
    parser.add_argument("--config-file", default="configs/mock-demo.json")
    parser.add_argument("--config-name", default=None, help="Run only this named configuration")
    parser.add_argument("--all", action="store_true", help="Run every configuration on its own port")
    parser.add_argument("--host", default=None, help="Override host (default: 127.0.0.1)")
    args = parser.parse_args()

    import uvicorn

    configs = load_configs(args.config_file)
    if args.config_name:
        configs = {args.config_name: configs[args.config_name]}

    def run(name: str, config: Dict[str, Any]) -> None:
        uvicorn.run(
            create_app(name, config),
            host=args.host or config.get("API_host", "127.0.0.1"),
            port=int(config.get("API_port", 8000)),
            log_level="info",
        )

    if args.all and len(configs) > 1:
        import multiprocessing

        jobs = [multiprocessing.Process(target=run, args=(n, c)) for n, c in configs.items()]
        for job in jobs:
            job.start()
        for job in jobs:
            job.join()
    else:
        name, config = next(iter(configs.items()))
        run(name, config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
