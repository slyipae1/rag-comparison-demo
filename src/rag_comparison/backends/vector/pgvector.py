import os
import sys
import time
import json
import argparse
import torch
import psycopg2
from psycopg2.extras import execute_values

from .base import VectorDB
from rag_comparison.model_utils.setup import setup_embedding_model
from rag_comparison.utils.data_loading import load_data
from rag_comparison.instrumentation.logging_utils import setup_logger
from rag_comparison.utils.db_logger import DBLogger

logger = setup_logger("pgvectorDB")

class pgvectorDB(VectorDB):
    """
    pgvectorDB: A VectorDB implementation using PostgreSQL + pgvector extension.
    """

    def __init__(self, config):
        self.config = config
        self.embedding_model = setup_embedding_model(self.config.get("embedding_model"))
        self.db_name = self.config.get("pgvector_db_name", "ragdb")
        self.user = self.config.get("pgvector_user", "postgres")
        self.password = self.config.get("pgvector_password", "postgres")
        self.host = self.config.get("pgvector_host", "localhost")
        self.port = self.config.get("pgvector_port", 5432)
        self.table_name = self.config.get("pgvector_table_name", "rag_vectors")
        self.embedding_dim = self.config.get("embedding_dim", 1024)
        self.conn = self._connect()
        self._ensure_table()

    def _connect(self):
        try:
            conn = psycopg2.connect(
                dbname=self.db_name,
                user=self.user,
                password=self.password,
                host=self.host,
                port=self.port
            )
            logger.info("Connected to pgvector database.")
            return conn
        except Exception as e:
            logger.error(f"Failed to connect to pgvector database: {e}")
            raise

    def _ensure_table(self):
        cur = self.conn.cursor()
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {self.table_name} (
                id SERIAL PRIMARY KEY,
                document TEXT,
                embedding vector({self.embedding_dim})
            );
        """)
        self.conn.commit()
        cur.close()
        logger.info(f"Ensured table {self.table_name} exists.")

    def add_documents(self, documents):
        """
        Add documents to the pgvector table with embeddings.
        """
        embeddings = self.embedding_model.embed_documents(documents)
        values = [(doc, emb) for doc, emb in zip(documents, embeddings)]
        cur = self.conn.cursor()
        execute_values(
            cur,
            f"INSERT INTO {self.table_name} (document, embedding) VALUES %s",
            values,
            template="(%s, %s)"
        )
        self.conn.commit()
        cur.close()
        logger.info(f"Inserted {len(documents)} documents into {self.table_name}.")

    def getDocs(self):
        """
        Returns all documents in the database.
        """
        cur = self.conn.cursor()
        cur.execute(f"SELECT document FROM {self.table_name}")
        docs = [row[0] for row in cur.fetchall()]
        cur.close()
        return docs

    def setup_db_retriever(self):
        """
        Returns a retriever object compatible with CompressionRetriever.
        """
        return pgvectorRetriever(self, self.config)

    def search(self, query, top_k=5):
        """
        Search for top_k most similar documents to the query.
        """
        query_emb = self.embedding_model.embed_query(query)
        # Ensure query_emb is a list of floats
        if hasattr(query_emb, "tolist"):
            query_emb = query_emb.tolist()
        # Format as a PostgreSQL array string
        emb_str = ",".join([str(float(x)) for x in query_emb])
        emb_sql = f"ARRAY[{emb_str}]::vector"
        cur = self.conn.cursor()
        cur.execute(
            f"""
            SELECT document, embedding <#> {emb_sql} AS distance
            FROM {self.table_name}
            ORDER BY distance ASC
            LIMIT %s
            """,
            (top_k,)
        )
        results = cur.fetchall()
        cur.close()
        return [row[0] for row in results]

from langchain_core.retrievers import BaseRetriever
from typing import Any

class pgvectorRetriever(BaseRetriever):
    """
    Simple retriever for pgvectorDB, compatible with CompressionRetriever and LangChain.
    """
    db: Any
    top_k: int

    def __init__(self, db: Any, config):
        super().__init__(db=db, top_k=config.get("retrieval_top_k", 5))

    def _get_relevant_documents(self, query):
        docs = self.db.search(query, top_k=self.top_k)
        from langchain_core.documents import Document
        return [Document(page_content=doc) for doc in docs]
    
    def batch(self, queries):
        # Application-level batch: call single query for each
        doc_batch = []
        return doc_batch.extend(self._get_relevant_documents(q) for q in queries)

########################################################################################
# Independent build and check utilities

import pickle

def build_pgvector_collection_from_pkl(config, all_text_data, conn, table_name):
    """
    Build pgvector DB from precomputed embeddings in a pickle file.
    The pickle should contain a dict with keys: 'doc_ids' and 'embeddings'.
    """
    embedding_pkl_path = config.get("embedding_pkl_path")
    logger.info(f"Loading embeddings from pickle: {embedding_pkl_path}")
    with open(embedding_pkl_path, "rb") as f:
        data = pickle.load(f)
    embeddings = data["embeddings"]
    ids = data.get("doc_ids", [str(i) for i in range(len(embeddings))])
    ids = [str(i) for i in ids]

    texts = all_text_data
    assert len(texts) == len(embeddings), "Number of texts and embeddings must match"

    cur = conn.cursor()
    cur.execute(f"TRUNCATE TABLE {table_name};")
    conn.commit()

    total = len(texts)
    batch_size = config.get("build_batch_size", 256)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_texts = texts[start:end]
        batch_embeddings = embeddings[start:end]
        # Convert numpy arrays to lists if needed
        batch_embeddings = [e.tolist() if hasattr(e, "tolist") else e for e in batch_embeddings]
        values = [(doc, emb) for doc, emb in zip(batch_texts, batch_embeddings)]
        execute_values(
            cur,
            f"INSERT INTO {table_name} (document, embedding) VALUES %s",
            values,
            template="(%s, %s)"
        )
        conn.commit()
        logger.info(f"Inserted documents {start} to {end} / {total}")
    cur.close()
    logger.info(f"Inserted {total} documents into {table_name} from pickle.")
    return True


def build_pgvector_db(embedding_model, text_data, conn, table_name, batch_size=1024):
    logger.info("Building pgvector DB...")
    cur = conn.cursor()
    cur.execute(f"TRUNCATE TABLE {table_name};")
    conn.commit()

    total = len(text_data)
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch_docs = text_data[start:end]
        with torch.no_grad():
            embeddings = embedding_model.embed_documents(batch_docs)
        values = [(doc, emb) for doc, emb in zip(batch_docs, embeddings)]
        execute_values(
            cur,
            f"INSERT INTO {table_name} (document, embedding) VALUES %s",
            values,
            template="(%s, %s)"
        )
        conn.commit()
        logger.info(f"Inserted documents {start} to {end} / {total}")

    cur.close()
    logger.info(f"Inserted {total} documents into {table_name}.")
    return True

def build_pgvector_db_with_multiple_files(config):
    """
    Build pgvector DB from multiple JSONL files or from a precomputed embedding pickle.
    """
    data_folder_path = config.get("data_dir", "./data")
    embedding_pkl_path = config.get("embedding_pkl_path")
    try:
        logger.info("Starting to build pgvector DB with multiple files")
        all_text_data = []
        db_logger = DBLogger("pgvectorDB_log.json")
        changed_files = db_logger.check_files_changed(data_folder_path, file_pattern="*.jsonl")
        if not changed_files:
            logger.info("No changes in JSONL files, skipping pgvector DB build")
            return None

        logger.info(f"Found {len(changed_files)} changed files: {changed_files}")
        for file in os.listdir(data_folder_path):
            if file.endswith(".jsonl"):
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
        db_name = config.get("pgvector_db_name", "ragdb")
        user = config.get("pgvector_user", "postgres")
        password = config.get("pgvector_password", "postgres")
        host = config.get("pgvector_host", "localhost")
        port = config.get("pgvector_port", 5432)
        table_name = config.get("pgvector_table_name", "rag_vectors")
        build_batch_size = config.get("build_batch_size", 512)

        # find embedding dim from embedding model
        embedding_model = setup_embedding_model(config.get("embedding_model"))
        try:
            dummy_text = ["This is a testing document for embedding dimension"]
            with torch.no_grad():
                dummy_embedding = embedding_model.embed_documents(dummy_text)
            if isinstance(dummy_embedding, list):
                embedding_dim = len(dummy_embedding[0])
            else:
                embedding_dim = dummy_embedding.shape[-1]
        except Exception as e:
            raise ValueError(f"Failed to get embedding dimension from model: {e}")
        del embedding_model
        print(f"embedding dimension: {embedding_dim}")
        if embedding_dim is None or embedding_dim <= 0:
            raise ValueError("Embedding model did not return a valid dimension. Please check the model setup.")

        conn = psycopg2.connect(
            dbname=db_name, user=user, password=password, host=host, port=port
        )
        cur = conn.cursor()
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS {table_name} (
                id SERIAL PRIMARY KEY,
                document TEXT,
                embedding vector({embedding_dim})
            );
        """)
        conn.commit()
        cur.close()

        if embedding_pkl_path and os.path.exists(embedding_pkl_path):
            logger.info(f"embedding_pkl_path found in config: {embedding_pkl_path}, building pgvector DB from pickle.")
            build_pgvector_collection_from_pkl(config, all_text_data, conn, table_name)
            logger.info("pgvector DB built from pickle.")
        else:
            embedding_model = setup_embedding_model(config.get("embedding_model"))
            build_pgvector_db(embedding_model, all_text_data, conn, table_name, build_batch_size)
            logger.info("pgvector DB built from text data.")

        db_logger.save_db_build_log("pgvectorDB", len(all_text_data))
        logger.info(f"pgvector DB created in {db_name}.{table_name} with {len(all_text_data)} total documents")
        conn.close()
        return True

    except Exception as e:
        logger.error(f"Error building pgvector DB: {str(e)}")
        raise

def check_pgvector_db(config):
    """
    Check if the pgvector DB is accessible and contains data.
    """
    try:
        db_name = config.get("pgvector_db_name", "ragdb")
        user = config.get("pgvector_user", "postgres")
        password = config.get("pgvector_password", "postgres")
        host = config.get("pgvector_host", "localhost")
        port = config.get("pgvector_port", 5432)
        table_name = config.get("pgvector_table_name", "rag_vectors")

        conn = psycopg2.connect(
            dbname=db_name, user=user, password=password, host=host, port=port
        )
        cur = conn.cursor()
        cur.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cur.fetchone()[0]
        logger.info(f"Table {table_name} contains {count} documents.")
        cur.execute(f"SELECT document FROM {table_name} LIMIT 1")
        sample = cur.fetchone()
        logger.info(f"Sample document: {sample[0] if sample else 'None'}")
        cur.close()
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Error checking pgvector DB: {str(e)}")
        return False