import torch
import os 

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_openai import ChatOpenAI

def setup_LLM(config):
    return ChatOpenAI(
            api_key=config.get("API_key", "EMPTY"),
            base_url=config.get('base_url', "http://localhost:30000/v1"),
            model= config.get('INTERNAL_MODEL', "DSF-LLM"),
            max_tokens = config.get('max_tokens', 4096),
            streaming= config.get('stream', False),
        )


def setup_embedding_model(embedding_model_name):

    return HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


def setup_rerank_model(rerank_model_name, reranker_batch_size=32):
    if "Qwen3-Reranker" in rerank_model_name:
        # Use Qwen3Reranker instead of HuggingFaceCrossEncoder
        from rag_comparison.model_utils.models.qwen3_reranker import Qwen3CrossEncoder, Qwen3Reranker
        qwen3 = Qwen3Reranker(
            model_path= rerank_model_name,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            reranker_batch_size = reranker_batch_size
        )
        # qwen3 = Qwen3Reranker(
        #     model_path=rerank_model_name,
        #     device= 'cpu'
        # )
        return Qwen3CrossEncoder(qwen3)
    else:
        return HuggingFaceCrossEncoder(
            model_name=rerank_model_name
        )
    