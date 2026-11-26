import asyncio

from rag_comparison.pipelines.mock import MockPipeline


def collect(config):
    pipeline = MockPipeline(config)

    async def run():
        return [chunk async for chunk in pipeline.process_question("What is RAG?")]

    return asyncio.run(run())


def test_llm_only_output_shape():
    chunks = collect({"RAG_pipeline_type": "mock_llm_only"})
    assert any("content" in c for c in chunks)
    assert any("reasoning" in c for c in chunks)
    assert any(isinstance(c.get("inference_time"), float) for c in chunks)


def test_retrieval_pipeline_is_distinguishable():
    llm_only = collect({"RAG_pipeline_type": "mock_llm_only"})
    rag = collect({"RAG_pipeline_type": "mock_rag"})
    assert "Synthetic retrieval" in rag[0]["reasoning"]
    assert "Synthetic retrieval" not in llm_only[0]["reasoning"]
