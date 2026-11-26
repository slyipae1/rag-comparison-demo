from abc import ABC, abstractmethod

class Retriever(ABC):
    @abstractmethod
    def invoke(self, query: str):
        pass

    @abstractmethod
    def batch(self, queries: list[str]):
        pass