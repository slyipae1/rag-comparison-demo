from fastapi.testclient import TestClient

from rag_comparison.api.app import create_app

CONFIG = {
    "RAG_pipeline_type": "mock_rag",
    "API_host": "127.0.0.1",
    "API_port": 8023,
    "mock_delay_seconds": 0.001,
}


def test_health_and_completion():
    with TestClient(create_app("mock", CONFIG)) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "healthy"

        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}]},
        )
        assert response.status_code == 200
        body = response.json()["choices"][0]
        assert body["message"]["content"]
        assert isinstance(body["inference_time"], float)


def test_streaming_completion():
    with TestClient(create_app("mock", CONFIG)) as client:
        response = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "Hello"}], "stream": True},
        )
        assert response.status_code == 200
        assert "data: [DONE]" in response.text
