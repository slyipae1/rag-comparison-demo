import torch
import torch.nn.functional as F
from typing import Any, Optional
from pydantic import BaseModel, Field
from langchain_core.embeddings import Embeddings

from transformers import AutoTokenizer, AutoModel


    
    # if "Qwen3-Embedding" in embedding_model_name:
    #     # Use Qwen3Embed for Qwen3-Embedding models
    #     from rag_comparison.model_utils.models.Qwen3Embed import Qwen3Embed
        # return Qwen3Embed(
        #     model_name=embedding_model_name,
        #     device='cuda:0' if torch.cuda.is_available() else 'cpu',
        #     normalize_embeddings=True
        # )

class Qwen3Embed(BaseModel, Embeddings):
    """
    Qwen3-Embedding-0.6B embedding model wrapper, compatible with LangChain DBs.
    """

    model_name: str = Field(default="Qwen/Qwen3-Embedding-0.6B", alias="model")
    model_path: Optional[str] = None
    device: str = Field(default="cuda:0" if torch.cuda.is_available() else "cpu")
    max_length: int = 8192
    normalize_embeddings: bool = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        model_path = self.model_path or self.model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        self.model = AutoModel.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

    @staticmethod
    def last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        # Tokenize
        batch_dict = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        batch_dict = {k: v.to(self.device) for k, v in batch_dict.items()}
        with torch.no_grad():
            outputs = self.model(**batch_dict)
            embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
            if self.normalize_embeddings:
                embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Compute embeddings for a list of documents.
        """
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        """
        Compute embedding for a single query.
        """
        return self._embed([text])[0]