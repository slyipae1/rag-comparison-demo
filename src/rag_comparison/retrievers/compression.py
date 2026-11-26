from rag_comparison.retrievers.base import Retriever
from rag_comparison.model_utils.setup import setup_rerank_model

from rag_comparison.instrumentation.logging_utils import setup_logger
logger = setup_logger("CompressionRetriever")

class CompressionRetriever(Retriever):
    def __init__(self, DB, config):
        super().__init__()
        self.config = config
        self.DB = DB
        self.rerank_model = setup_rerank_model(config.get("reranker_model", ""), config.get("reranker_batch_size", 32)) # "Qwen/Qwen3-Reranker-0.6B"
        self.bm25_retriever = self.setup_bm25_retriever()

        self.RetrievalInterface = self.as_retriever()

    def invoke(self, query):
        docs = self.RetrievalInterface.invoke(query)
        return docs
    
    def batch(self, queries):
        # Use the underlying retriever's batch method if available
        if hasattr(self.RetrievalInterface, "batch"):
            return self.RetrievalInterface.batch(queries)
        else:
            logger.info("Fallback to invoking queries one by one")
            return [self.invoke(q) for q in queries]
    
    def as_retriever(self):
        from langchain.retrievers.document_compressors import CrossEncoderReranker
        from langchain.retrievers import EnsembleRetriever, ContextualCompressionRetriever

        ensemble = EnsembleRetriever(
                    retrievers=[self.bm25_retriever, self.DB.setup_db_retriever()],
                    weights=[0.7, 0.3]
                )

        compressor = CrossEncoderReranker(
            model=self.rerank_model,
            top_n=self.config.get("rerank_top_k", 20)
        )
        return ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble
        )

    def setup_bm25_retriever(self):
        from langchain_community.retrievers import BM25Retriever
        from rag_comparison.instrumentation.logging_utils import setup_logger
        logger = setup_logger("retriever_setup")

        """从DB创建BM25检索器"""
        logger.info("Creating BM25 retriever from DB...")
        try:
            
            # 提取文档文本
            texts = self.DB.getDocs()
            
            if not texts:
                logger.warning("No documents found in DB")
                return None
                
            logger.info(f"Found {len(texts)} documents in DB")
            
            # 创建BM25检索器
            return BM25Retriever.from_texts(
                texts,
                search_kwargs={
                    "k": self.config.get("retrieval_top_k", 50) * 2,
                    "b": 0.75,
                    "k1": 1.5
                }
            )
        except Exception as e:
            logger.error(f"Error creating BM25 retriever: {str(e)}")
            raise
