import time
import os
import sys
import subprocess
import signal

import pickle

from .base import VectorDB
from langchain_chroma import Chroma
from chromadb.config import Settings as ChromaSettings
import chromadb
from chromadb.config import Settings

from rag_comparison.model_utils.setup import setup_embedding_model
from rag_comparison.utils.data_loading import load_data
from rag_comparison.instrumentation.logging_utils import setup_logger
from rag_comparison.utils.db_logger import DBLogger
logger = setup_logger("chromaDB_build")

class ChromaDB(VectorDB):
    def __init__(self, config):
        self.config = config
        self.embedding_model = setup_embedding_model(self.config.get("embedding_model", "qWEN/Qwen3-Embedding-0.6B"))
        self.persist_directory = self.config.get("persist_directory", "./db/chromaDB")
        self.collection_name = self.config.get("collection_name", "RAG")

        if not config.get("use_server_mode", False): # use local persisit mode
            logger.info("Loading Existing Chroma index...")
            logger.debug(f"Persist directory: {self.persist_directory}")
            db = Chroma(
                persist_directory=self.persist_directory,
                collection_name=self.collection_name,
                embedding_function=self.embedding_model,
            )
            logger.info(" Local Chroma index loaded")
            count = db._collection.count()
            logger.info(f"Collection contains {count} documents")
            
            self.db = db
        else: # use server mode
            """设置 Chroma 数据库服务器连接"""
            try:
                logger.info("Setting up Chroma DB server connection")
                
                # 配置客户端设置
                
                cdb_settings = ChromaSettings(
                    chroma_server_host=self.config.get("chroma_server_host", "localhost"),
                    chroma_server_http_port=self.config.get("chroma_server_port", 8030),
                    chroma_server_ssl_enabled=True,
                    anonymized_telemetry=True,
                )
                
                # 创建客户端连接

                client = chromadb.HttpClient(cdb_settings.chroma_server_host, port = cdb_settings.chroma_server_http_port)

                # 等待服务器准备就绪
                max_retries = 3
                for i in range(max_retries):
                    try:
                        client.heartbeat()
                        logger.info("Successfully connected to Chroma server")
                        break
                    except Exception as e:
                        if i == max_retries - 1:
                            raise
                        logger.warning(f"Retry {i+1}/{max_retries} connecting to server")
                        time.sleep(2)
                
                # 初始化 Chroma
                db = Chroma(
                    client=client,
                    collection_name=self.collection_name,
                    embedding_function=self.embedding_model,
                    persist_directory=self.persist_directory,
                )
            
                count = db._collection.count()
                logger.info(f"Collection contains {count} documents")
                
                logger.info("Server Chroma index loaded")
                self.db = db
            except Exception as e:
                logger.error(f"Error setting up Chroma DB server: {str(e)}", exc_info=True)
                raise

    def getDocs(self):
        """
        Returns all documents in the database.
        """
        collection = self.db.get()
        texts = collection['documents'] if collection and 'documents' in collection else []

        return texts

    def setup_db_retriever(self):
        """
        Should return a DB retriever object that can be used in setup_ensemble_retriever.
        """
        return self.db.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": self.config.get("retrieval_top_k", 5),
            "fetch_k": self.config.get("retrieval_fetch_k", 10),
            "lambda_mult": 0.5
        }
    )


########################################################################################
# indenpendent build and check

def build_chroma_db_from_pkl(config, all_text_data):
    """
    Build a Chroma DB collection from precomputed embeddings in a pickle file.
    The pickle should contain a dict with keys: 'doc_ids' and 'embeddings'.
    """
    embedding_pkl_path = config.get("embedding_pkl_path")
    persist_directory = config.get("persist_directory", "./db/chromaDB")
    collection_name = config.get("collection_name", "RAG")
    batch_size = config.get("build_batch_size", 512)

    logger.info(f"Loading embeddings from pickle: {embedding_pkl_path}")
    with open(embedding_pkl_path, "rb") as f:
        data = pickle.load(f)
    embeddings = data["embeddings"]
    ids = data.get("doc_ids", [str(i) for i in range(len(embeddings))])
    ids = [str(i) for i in ids]

    texts = all_text_data
    assert len(texts) == len(embeddings), "Number of texts, embeddings, and ids must match"

    db = Chroma(
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_function=None,  # Not needed since we have embeddings
    )

    total = len(texts)
    for i in range(0, total, batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_embeddings = embeddings[i:i+batch_size]
        batch_ids = ids[i:i+batch_size]
        logger.info(f"Adding batch {i} - {i+len(batch_texts)-1} / {total}")
        db._collection.upsert(
            embeddings=batch_embeddings,
            documents=batch_texts,
            ids=batch_ids,
        )

    return db, persist_directory

def build_chroma_db(embedding_model, text_data, persist_directory, collection_name):

    logger.info("Creating Chroma index...")
    logger.debug(f"Persist directory: {persist_directory}")
    
    db = Chroma.from_texts(
        texts=text_data,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    logger.info("Chroma index created")
    logger.info(f"db: {db}")
    # logger.info(f"db.count(): {db.count()}")
    
    return db, persist_directory

def build_chroma_db_with_multiple_files(config):

    data_folder_path = config.get("data_dir", "./data")
    logger.info("Starting to build Chroma DB with multiple files")
    all_text_data = []
    db_logger = DBLogger("ChromaDB_log.json")
    changed_files = db_logger.check_files_changed(data_folder_path, file_pattern="*.jsonl")
    if not changed_files:
        logger.info("No changes in JSONL files, skipping Chroma DB build")
        return None, None

    logger.info(f"Found {len(changed_files)} changed files: {changed_files}")
    logger.info("Building Chroma DB with changed files")

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
        return None, None

    logger.info(f"Total documents loaded: {len(all_text_data)}")

    """
    Build a Chroma DB collection from all JSONL files in the data directory,
    or from a precomputed embedding pickle if specified in config.
    """
    # remove old DB if exists
    if os.path.exists(config.get("persist_directory", "./db/chromaDB")):
        logger.info(f"Removing old Chroma DB at {config.get('persist_directory', './db/chromaDB')}")
        os.system(f"rm -rf {config.get('persist_directory', './db/chromaDB')}")
    os.mkdir(config.get("persist_directory", "./db/chromaDB"))

    embedding_pkl_path = config.get("embedding_pkl_path")
    if embedding_pkl_path and os.path.exists(embedding_pkl_path):
        logger.info(f"embedding_pkl_path found in config: {embedding_pkl_path}, building Chroma DB from pickle.")
        db, persist_directory = build_chroma_db_from_pkl(config, all_text_data)
        logger.info("Chroma DB built from pickle.")

    else:
        embedding_model = setup_embedding_model(config.get("embedding_model", "qWEN/Qwen3-Embedding-0.6B"))
        persist_directory = config.get("persist_directory", "./db/chromaDB")
        collection_name = config.get("collection_name", "RAG")
        db, persist_directory = build_chroma_db(embedding_model, all_text_data, persist_directory, collection_name)
        logger.info("Chroma DB built from text data.")

    db_logger.save_db_build_log("ChromaDB", len(all_text_data))
    logger.info(f"Chroma index created in {persist_directory} with {len(all_text_data)} total documents")

    return db, persist_directory

def run_chroma_server(config):
    """启动ChromaDB服务器"""
    try:
        
        # 构建命令
        cmd = [
            "chroma",
            "run",
            "--path", config['persist_directory'],
            "--host", config['chroma_server_host'],
            "--port", str(config['chroma_server_port'])  # Ensure port is a string
        ]
        
        logger.info(f"启动命令: {' '.join(cmd)}")
        
        # 启动服务器进程
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        # 处理Ctrl+C信号
        def signal_handler(sig, frame):
            logger.info("正在关闭ChromaDB服务器...")
            process.terminate()
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        
        # 实时输出日志
        while True:
            output = process.stdout.readline()
            if output:
                print(output.strip())
            
            error = process.stderr.readline()
            if error:
                print(f"错误: {error.strip()}", file=sys.stderr)
            
            # 检查进程是否还在运行
            if process.poll() is not None:
                break
        
        return_code = process.poll()
        if return_code != 0:
            logger.error(f"ChromaDB服务器异常退出，返回码: {return_code}")
            return False
            
    except Exception as e:
        logger.error(f"启动ChromaDB服务器时出错: {str(e)}")
        return False
    
    return True

def verify_server_data(config):
    try:
        # 连接到服务器
        server_client = chromadb.HttpClient(
            host=config.get("chroma_server_host", "localhost"),
            port=config.get("chroma_server_port", 8030),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # 列出所有集合
        collection_names = server_client.get_collection(config.get("collection_name", "RAG"))
        logger.info(f"服务器中的集合列表: {collection_names}")
        
        try:
            # 直接尝试获取集合
            collection = server_client.get_collection(config.get("collection_name", "RAG"))  # 替换为你的集合名称
            count = collection.count()
            logger.info(f"集合中的文档数量: {count}")
            
            # 获取样本数据进行验证
            sample = collection.get(limit=1)
            logger.info(f"样本数据: {sample}")
            return True
        except Exception as e:
            logger.error(f"获取集合 '{os.getenv('CHROMA_COLLECTION_NAME')}' 失败: {str(e)}")
            return False
        

    except Exception as e:
        logger.error(f"验证服务器数据时出错: {str(e)}")
        return False
