from abc import ABC, abstractmethod


class VectorDB(ABC):
    @abstractmethod
    def setup_db_retriever(self):
        """
        Should return a DB retriever object that can be used in setup_ensemble_retriever.
        """
        pass
    @abstractmethod
    def getDocs(self):
        """
        Should return a list of all documents in the database.
        """
        pass



# recommended for independent build and check
# def build_db_with_multiple_files(config):
#     pass

# def check_db(config):
#     pass
