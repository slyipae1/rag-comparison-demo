from abc import ABC, abstractmethod

class RAGpipeline(ABC):
    @abstractmethod
    async def process_question(self, question: str, use_stream: bool = False):
        """
        Process a question using the RAG pipeline.

        Args:
            question (str): The input question to process.
            use_stream (bool, optional): Whether to use streaming. Defaults to False.

        Returns:
            Any: The result of processing the question.
        """
        pass