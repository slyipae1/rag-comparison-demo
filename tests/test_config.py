import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_mock_config_is_local_and_offline():
    cfg = json.loads((ROOT / "configs/mock-demo.json").read_text())
    assert len(cfg) >= 2
    for name, value in cfg.items():
        assert value["API_host"] == "127.0.0.1", name
        assert value["RAG_pipeline_type"].startswith("mock_"), name


def test_example_config_uses_local_defaults():
    cfg = json.loads((ROOT / "configs/example.json").read_text())
    for name, value in cfg.items():
        assert value["API_host"] == "127.0.0.1", name
        for key in ("internal_LLM", "external_LLM"):
            assert value[key]["base_url"].startswith("http://localhost:"), name
            assert value[key]["API_key"] == "EMPTY", name

    db = cfg["LLM with Google & DB (MilvusDB)"]
    assert db["milvus_host"] == "localhost"
    assert db["data_dir"].startswith("./")

    graph = cfg["LLM with Google & Graph RAG (HiRAG)"]
    assert graph["data_dir"].startswith("./")
    assert graph["hirag"]["working_dir"].startswith("./")
