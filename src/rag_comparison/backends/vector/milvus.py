import os
import sys
import time
import json
import argparse
import logging
import torch

from .base import VectorDB

from pymilvus import (
    connections, Collection, CollectionSchema, FieldSchema, DataType,
    utility, MilvusException
)
from langchain_community.vectorstores import Milvus as LangchainMilvus

from rag_comparison.model_utils.setup import setup_embedding_model
from rag_comparison.utils.data_loading import load_data
from rag_comparison.instrumentation.logging_utils import setup_logger

logger = setup_logger("milvusDB_build")

class MilvusDB(VectorDB):
    def __init__(self, config):
        self.config = config
        self.embedding_model = setup_embedding_model(self.config.get("embedding_model", "Qwen/Qwen3-Embedding-0.6B"))
        self.collection_name = self.config.get("collection_name", "RAG")
        self.host = self.config.get("milvus_host", "localhost")
        self.port = self.config.get("milvus_port", "19530")
        self.user = self.config.get("milvus_user", None)
        self.password = self.config.get("milvus_password", None)
        self.db_name = self.config.get("milvus_db_name", "default")
        self.dim = self.config.get("embedding_dim", 1024)
        self.index_type = self.config.get("milvus_index_type", "IVF_FLAT")
        self.metric_type = self.config.get("milvus_metric_type", "L2")
        self.top_k = self.config.get("retrieval_top_k", 5)
        self.persist_directory = self.config.get("persist_directory", "./db/milvusDB")

        # Connect to Milvus
        try:
            logger.info(f"Connecting to Milvus at {self.host}:{self.port} (db: {self.db_name})")
            connections.connect(
                alias="default",
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                db_name=self.db_name
            )
            logger.info("Connected to Milvus successfully.")
        except Exception as e:
            logger.error(f"Failed to connect to Milvus: {str(e)}")
            raise

        # Check if collection exists
        if not utility.has_collection(self.collection_name):
            logger.error(f"Collection '{self.collection_name}' does not exist. Please build the collection first.")
            raise RuntimeError(f"Collection '{self.collection_name}' does not exist in Milvus.")

        self.collection = Collection(self.collection_name)
        logger.info(f"Loaded Milvus collection: {self.collection_name}")

        # Setup langchain Milvus wrapper
        self.db = LangchainMilvus(
            embedding_function=self.embedding_model,
            collection_name=self.collection_name,
            connection_args={
                "host": self.host,
                "port": self.port,
                "user": self.user,
                "password": self.password,
                "db_name": self.db_name
            }
        )
        logger.info(f"Loaded Milvus collection: {self.collection_name}")
        if self.collection.indexes:
            logger.info(f"Collection has {len(self.collection.indexes)} indexes.")
            for idx in self.collection.indexes:
                logger.info(f"Index object: {idx}")
                logger.info(f"Index params: {getattr(idx, 'params', 'No params attribute')}")
                logger.info(f"Index info (dict): {idx.to_dict() if hasattr(idx, 'to_dict') else 'No to_dict()'}")
        else:
            logger.info("No indexes found in collection.")

    def getDocs(self):
        """
        Returns all documents in the Milvus collection.
        """
        try:
            expr = ""  # get all
            output_fields = ["text"]
            res = self.collection.query(expr, output_fields=output_fields, limit=16384)  # Add a large limit or make it configurable
            texts = [item["text"] for item in res if "text" in item]
            return texts
        except Exception as e:
            logger.error(f"Error fetching documents from Milvus: {str(e)}")
            return []

    def setup_db_retriever(self):
        """
        Returns a retriever object compatible with langchain.
        """
        k = self.config.get("retrieval_top_k", 5)
        # Use nprobe for IVF_FLAT, ef for HNSW
        search_kwargs = {
            "k": k,
            "metric_type": self.metric_type,
        }

        if self.index_type.upper() == "IVF_FLAT":
            search_kwargs["nprobe"] = max(k * 2, 64)
        elif self.index_type.upper() == "HNSW":
            search_kwargs["ef"] = max(k * 2, 64)
        
        logger.info(f"Milvus search_kwargs: {search_kwargs}")
        return self.db.as_retriever(
            search_type="similarity",
            search_kwargs=search_kwargs
        )

########################################################################################
# Independent build and check
from rag_comparison.utils.db_logger import DBLogger

def truncate_by_bytes(text, max_bytes=65535):
    encoded = text.encode("utf-8")
    if len(encoded) <= max_bytes:
        return text
    # Truncate by bytes, then decode safely
    truncated = encoded[:max_bytes]
    # Remove incomplete trailing character if any
    while True:
        try:
            return truncated.decode("utf-8")
        except UnicodeDecodeError:
            truncated = truncated[:-1]

def build_milvus_collection_from_pkl(config, all_text_data):
    """
    Build a Milvus collection from precomputed embeddings in a pickle file.
    The pickle should contain a dict with keys: 'doc_ids' and 'embeddings'.
    """
    import pickle

    embedding_pkl_path = config.get("embedding_pkl_path")
    collection_name = config.get("collection_name", "RAG")
    host = config.get("milvus_host", "localhost")
    port = config.get("milvus_port", "19530")
    user = config.get("milvus_user", None)
    password = config.get("milvus_password", None)
    db_name = config.get("milvus_db_name", "default")
    index_type = config.get("milvus_index_type", "IVF_FLAT")
    metric_type = config.get("milvus_metric_type", "L2")
    batch_size = config.get("build_batch_size", 1024)

    logger.info(f"Loading embeddings from pickle: {embedding_pkl_path}")
    with open(embedding_pkl_path, "rb") as f:
        data = pickle.load(f)
    embeddings = data["embeddings"]
    dim = len(embeddings[0]) if embeddings else 0   
    print(f"embedding dimension: {dim}")
    if dim == 0:
        raise ValueError("No embeddings found in the pickle file.")
    ids = data.get("doc_ids", [str(i) for i in range(len(embeddings))])
    ids = [str(i) for i in ids]

    texts = all_text_data
    assert len(texts) == len(embeddings), "Number of texts and embeddings must match"

    # Connect
    connections.connect(
        alias="default",
        host=host,
        port=port,
        user=user,
        password=password,
        db_name=db_name
    )

    # Drop if exists
    if utility.has_collection(collection_name):
        logger.info(f"Dropping existing collection: {collection_name}")
        utility.drop_collection(collection_name)

    # Define schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields, description="RAG Collection")

    # Create collection
    collection = Collection(collection_name, schema)
    logger.info(f"Created collection: {collection_name}")

    # Insert data in batches
    n = len(texts)
    logger.info(f"Inserting {n} documents into Milvus in batches of {batch_size}...")
    for i in range(0, n, batch_size):
        batch_texts = texts[i:i+batch_size]
        # Truncate texts that are too long
        max_text_len = 65535
    for i in range(0, n, batch_size):
        batch_texts = texts[i:i+batch_size]
        temp = batch_texts.copy()
        batch_texts = [
            truncate_by_bytes(t, max_text_len) for t in batch_texts
        ]
        if temp != batch_texts:
            logger.warning(f"Truncated texts by max {max_text_len} bytes.")
        batch_embeddings = embeddings[i:i+batch_size]
        # if i//batch_size + 1 == 1882:
        #     print('\n\n'.join(batch_texts))
        collection.insert([batch_texts, batch_embeddings])
        logger.info(f"Inserted batch {i//batch_size + 1} ({i} to {min(i+batch_size, n)})")

    collection.flush()
    logger.info("Data inserted and flushed.")

    # Create index
    logger.info("Creating index...")
    if index_type.upper() == "IVF_FLAT":
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": metric_type,
            "params": {"nlist": 128}
        }
    elif index_type.upper() == "HNSW":
        index_params = {
            "index_type": "HNSW",
            "metric_type": metric_type,
            "params": {"M": 8, "efConstruction": 64}
        }
    else:
        raise ValueError(f"Unsupported index_type: {index_type}")
    collection.create_index(field_name="vector", index_params=index_params)
    logger.info("Index created.")

    return collection



def build_milvus_collection(embedding_model, text_data, config):
    """
    Build a Milvus collection and insert data in batches.
    """
    collection_name = config.get("collection_name", "RAG")
    host = config.get("milvus_host", "localhost")
    port = config.get("milvus_port", "19530")
    user = config.get("milvus_user", None)
    password = config.get("milvus_password", None)
    db_name = config.get("milvus_db_name", "default")
    index_type = config.get("milvus_index_type", "IVF_FLAT")
    metric_type = config.get("milvus_metric_type", "L2")
    batch_size = config.get("build_batch_size", 1024)  # Add this line

    # Connect
    connections.connect(
        alias="default",
        host=host,
        port=port,
        user=user,
        password=password,
        db_name=db_name
    )

    # Drop if exists
    if utility.has_collection(collection_name):
        logger.info(f"Dropping existing collection: {collection_name}")
        utility.drop_collection(collection_name)

    # find embedding dim from embedding model
    try:
        dummy_text = ["This is a testing document for embedding dimension"]
        with torch.no_grad():
            dummy_embedding = embedding_model.embed_documents(dummy_text)
        if isinstance(dummy_embedding, list):
            dim = len(dummy_embedding[0])
        else:
            dim = dummy_embedding.shape[-1]
    except Exception as e:
        raise ValueError(f"Failed to get embedding dimension from model: {e}")
    print(f"embedding dimension: {dim}")
    if dim is None or dim <= 0:
        raise ValueError("Embedding model did not return a valid dimension. Please check the model setup.")

    # Define schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535), # miluvs limit of max_length is 65535
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields, description="RAG Collection")

    # Create collection
    collection = Collection(collection_name, schema)
    logger.info(f"Created collection: {collection_name}")

    # Insert data in batches
    texts = text_data
    n = len(texts)
    
    logger.info(f"Inserting {n} documents into Milvus in batches of {batch_size}...")
    for i in range(0, n, batch_size):
        batch_texts = texts[i:i+batch_size]
        # Truncate texts that are too long
        max_text_len = 65535
    for i in range(0, n, batch_size):
        batch_texts = texts[i:i+batch_size]
        temp = batch_texts.copy()
        batch_texts = [
            truncate_by_bytes(t, max_text_len) for t in batch_texts
        ]
        if temp != batch_texts:
            logger.warning(f"Truncated texts by max {max_text_len} bytes.")
        with torch.no_grad():
            batch_embeddings = embedding_model.embed_documents(batch_texts)
        collection.insert([batch_texts, batch_embeddings])
        logger.info(f"Inserted batch {i//batch_size + 1} ({i} to {min(i+batch_size, n)})")

    collection.flush()
    logger.info("Data inserted and flushed.")

    # Create index
    logger.info("Creating index...")
    if index_type.upper() == "IVF_FLAT":
        index_params = {
            "index_type": "IVF_FLAT",
            "metric_type": metric_type,
            "params": {"nlist": 128}
        }
    elif index_type.upper() == "HNSW":
        index_params = {
            "index_type": "HNSW",
            "metric_type": metric_type,
            "params": {"M": 8, "efConstruction": 64}
        }
    else:
        raise ValueError(f"Unsupported index_type: {index_type}")
    collection.create_index(field_name="vector", index_params=index_params)
    logger.info("Index created.")

    return collection

def build_milvus_db_with_multiple_files(config):
    """
    Build Milvus DB from multiple JSONL files or from a precomputed embedding pickle if specified.
    Uses DBLogger to skip rebuild if no changes detected.
    """
    data_folder_path = config.get("data_dir", "./data")
    try:
        logger.info("Starting to build Milvus DB with multiple files")
        all_text_data = []
        db_logger = DBLogger("MilvusDB_log.json")
        changed_files = db_logger.check_files_changed(data_folder_path, file_pattern="*.jsonl")
        if not changed_files:
            logger.info("No changes in JSONL files, skipping Milvus DB build")
            return None

        logger.info(f"Found {len(changed_files)} changed files: {changed_files}")
        logger.info("Building Milvus DB with changed files")

        # Collect all JSONL JSON file data
        for file in sorted(os.listdir(data_folder_path)):
            if file.endswith(".jsonl") or file.endswith(".json"):
                data_path = os.path.join(data_folder_path, file)
                logger.info(f"Loading data from {data_path}")
                file_text_data = load_data(data_path)
                all_text_data.extend(file_text_data)
                logger.info(f"Loaded {len(file_text_data)} documents from {file}")
                db_logger.update_file_record(data_path, len(file_text_data))

        if not all_text_data:
            logger.warning("No data found in JSONL files")
            return None

        logger.info(f"Total documents loaded: {len(all_text_data)}")

        embedding_pkl_path = config.get("embedding_pkl_path")
        if embedding_pkl_path and os.path.exists(embedding_pkl_path):
            logger.info(f"embedding_pkl_path found in config: {embedding_pkl_path}, building Milvus DB from pickle.")
            collection = build_milvus_collection_from_pkl(config, all_text_data)
            logger.info("Milvus DB built from pickle.")
        else:
            embedding_model = setup_embedding_model(config.get("embedding_model", "Qwen/Qwen3-Embedding-0.6B"))
            collection = build_milvus_collection(embedding_model, all_text_data, config)
            logger.info(f"Milvus collection '{config.get('collection_name', 'RAG')}' built with {len(all_text_data)} documents")

        db_logger.save_db_build_log("MilvusDB", len(all_text_data))
        logger.info(f"Milvus index created in {config.get('persist_directory', './db/milvusDB')} with {len(all_text_data)} total documents")

        return collection

    except Exception as e:
        logger.error(f"Error building Milvus DB: {str(e)}")
        raise

    except Exception as e:
        logger.error(f"Error building Milvus DB: {str(e)}")
        raise

def check_milvus_server(config):
    """
    Check if Milvus server is running and collection exists.
    """
    try:
        host = config.get("milvus_host", "localhost")
        port = config.get("milvus_port", "19530")
        user = config.get("milvus_user", None)
        password = config.get("milvus_password", None)
        db_name = config.get("milvus_db_name", "default")
        collection_name = config.get("collection_name", "RAG")

        connections.connect(
            alias="default",
            host=host,
            port=port,
            user=user,
            password=password,
            db_name=db_name
        )
        if utility.has_collection(collection_name):
            logger.info(f"Collection '{collection_name}' exists in Milvus.")
            collection = Collection(collection_name)
            logger.info(f"Collection '{collection_name}' has {collection.num_entities} entities.")
            return True
        else:
            logger.error(f"Collection '{collection_name}' does not exist.")
            return False
    except Exception as e:
        logger.error(f"Error checking Milvus server: {str(e)}")
        return False

def check_server_avail(config):
    """
    Check if Milvus server is available (connectivity only).
    """
    from pymilvus import connections
    host = config.get("milvus_host", "localhost")
    port = config.get("milvus_port", "19530")
    user = config.get("milvus_user", None)
    password = config.get("milvus_password", None)
    db_name = config.get("milvus_db_name", "default")
    try:
        connections.connect(
            alias="default",
            host=host,
            port=port,
            user=user,
            password=password,
            db_name=db_name
        )
        return True
    except Exception as e:
        logger.error(f"Error checking Milvus server: {str(e)}")
        return False
