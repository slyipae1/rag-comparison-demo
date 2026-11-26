# Architecture

This project is an engineering demo, not a new retrieval algorithm. The same chat
request is routed to one or more configured pipelines; each pipeline emits
answer text, an optional reasoning/retrieval trace and stage timings.

```text
                 ┌────────────────────────┐
  question ─────▶│  FastAPI /v1/chat/...  │
                 └───────────┬────────────┘
                             │ RAG_pipeline_type
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
  MockPipeline        LLM_Google          LLM_Google_DB / LLM_Google_Graph_RAG
  (offline)           (web search)        (web + vector DB / graph RAG)
        │                    │                    │
        └────────────────────┼────────────────────┘
                             ▼
             {content, reasoning, inference_time}
                             │
                             ▼
                 Gradio side-by-side UI
```

## Components

- `src/rag_comparison/api/app.py`: OpenAI-style endpoint, one app per named
  configuration. Binds to `127.0.0.1` unless explicitly overridden.
- `src/rag_comparison/pipelines/`: abstract interface plus three augmented
  configurations and two deterministic mock configurations.
- `src/rag_comparison/retrievers/`: web-search (SerpAPI) and reranking wrappers.
- `src/rag_comparison/backends/vector/`: optional Milvus, Chroma and pgvector
  adapters.
- `src/rag_comparison/backends/graph/`: optional HiRAG, LightRAG and HippoRAG
  adapters. These require external packages.
- `src/rag_comparison/instrumentation/`: local logging and timing recorders.
- `ui/app.py`: lightweight Gradio UI for side-by-side comparison.

## Design notes

- The two augmented pipelines run web retrieval and local retrieval
  **concurrently** with `asyncio`, yielding whichever trace finishes first in
  streaming mode.
- `inference_time` in the API response is total wall-clock time for the
  pipeline invocation. Per-stage times can be written to a local JSONL file
  when `record_timing` is enabled.
- Configuration drives pipeline selection instead of hard-coded branches in
  the UI. The UI simply reads the endpoint list from the config file.
