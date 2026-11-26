from abc import ABC, abstractmethod

class Graph_RAG(ABC):
    @abstractmethod
    async def initialize(self):
        """
        Initialize the RAG pipeline or storages.
        """
        pass

    @abstractmethod
    def index(self, docs):
        """
        Index documents into the RAG system.
        """
        pass

    @abstractmethod
    def retrieve(self, query, retrieval_top_k=5, parsed=None):
        """
        Retrieve documents or answers for a query.
        """
        pass

    @abstractmethod
    def retrieve_batch(self, queries, retrieval_top_k=5):
        """
        Retrieve documents or answers for a batch of queries.
        """
        pass

    @abstractmethod
    async def construct(self, id, docs, mode):
        """
        Construct or update the knowledge graph for a given id.
        """
        pass

    @abstractmethod
    def updateNeo4j(self, id):
        """
        Update the Neo4j database for a given knowledge graph id.
        """
        pass

    @abstractmethod
    def delete_KG(self, id):
        """
        Delete the knowledge graph for a given id.
        """
        pass