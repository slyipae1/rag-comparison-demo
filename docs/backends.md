# Optional backends

`configs/mock-demo.json` runs offline and needs no services. The configurations
in `configs/example.json` describe additional backends; provide the
corresponding services and credentials yourself before using them.

## Requirements by configuration

| Configuration | Needs |
|---|---|
| `LLM with Google` | An OpenAI-compatible model endpoint and a SerpAPI key (`SERPAPI_API_KEY`). |
| `LLM with Google & DB (MilvusDB)` | The above plus an embedding model, a reranker and a [Milvus](https://github.com/milvus-io/milvus) instance. |
| `LLM with Google & Graph RAG (HiRAG)` | The above plus [HiRAG](https://github.com/HKUDS/HiRAG), an embedding API and [Neo4j](https://github.com/neo4j/neo4j) if graph writes are enabled. |

## Upstream projects

| Category | Project | Link |
|---|---|---|
| Graph RAG | HiRAG | https://github.com/HKUDS/HiRAG |
| Graph RAG | LightRAG | https://github.com/HKUDS/LightRAG |
| Graph RAG | HippoRAG | https://github.com/OSU-NLP-Group/HippoRAG |
| Graph storage | Neo4j | https://github.com/neo4j/neo4j |
| Vector DB | Milvus | https://github.com/milvus-io/milvus |
| Vector DB | Chroma | https://github.com/chroma-core/chroma |
| Vector DB | pgvector | https://github.com/pgvector/pgvector |
| Arena UI reference | FastChat | https://github.com/lm-sys/FastChat |
| LLM tooling | LangChain | https://github.com/langchain-ai/langchain |

Adapters for these projects are included as references in
`src/rag_comparison/backends/`. Their upstream packages and servers are not
bundled here. Install only what you need with
`pip install -r requirements-full.txt`, follow each project's own installation
instructions, and never commit `.env` files containing real keys.

See [`NOTICE.md`](../NOTICE.md) for license attributions.
