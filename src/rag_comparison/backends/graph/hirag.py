from .base import Graph_RAG

import os
import shutil
import re
import pandas as pd
import io

from rag_comparison.utils.data_loading import load_data
from rag_comparison.instrumentation.logging_utils import setup_logger
from rag_comparison.utils.db_logger import DBLogger
logger = setup_logger("HiRAG")

import numpy as np
import yaml
from hirag import QueryParam
from hirag import HiRAG as HiRAG_pipeline

import xml.etree.ElementTree as ET
from rag_comparison.utils.neo4j_loader import NEO4JLOADER, NEO4J_SERVER_URI

import httpx
from dataclasses import dataclass
@dataclass
class EmbeddingFunc:
    embedding_dim: int
    max_token_size: int
    func: callable

    async def __call__(self, *args, **kwargs) -> np.ndarray:
        return await self.func(*args, **kwargs)

def wrap_embedding_func_with_attrs(**kwargs):
    def final_decro(func) -> EmbeddingFunc:
        return EmbeddingFunc(**kwargs, func=func)
    return final_decro

class HiRAG(Graph_RAG):
    def __init__(self, config):
        self.config = config
        llm_config = self.config.get("internal_LLM")

        # define model functions
        @wrap_embedding_func_with_attrs(
            embedding_dim=config['embedding_model_api']['embedding_dim'],
            max_token_size=config['embedding_model_api'].get('max_token_size', 8192)  # default to 8192 if not specified
        )
        async def _embedding(texts: list[str]) -> np.ndarray:
            url = config["embedding_model_api"]["base_url"]
            headers = {"authorization": config["embedding_model_api"]["API_key"]}
            payload = {"input": texts}
            async with httpx.AsyncClient(timeout=90.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()["data"]
                return np.array([d["embedding"] for d in data])
            
        self._embedding = _embedding

        async def _model(prompt, system_prompt=None, history_messages=[], **kwargs):
            url = config["internal_LLM"]["base_url"]+ "/chat/completions"
            headers = {"authorization": config["internal_LLM"]["API_key"]}
            payload = {
                "model": config["internal_LLM"]["model_name"],
                "messages": [{"role": "user", "content": prompt}],
            }
            async with httpx.AsyncClient(timeout=9000.0) as client:
                resp = await client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
        
        self._model = _model

        if "all_collection_dir" in config:
            self.all_collection_dir = config["all_collection_dir"]
        else:
            self.all_collection_dir = "./collections_HiRAG"
        if not os.path.exists(self.all_collection_dir):
            os.makedirs(self.all_collection_dir, exist_ok=True)

        self.all_id2collection = {}
        for id in os.listdir(self.all_collection_dir):
            if os.path.isdir(os.path.join(self.all_collection_dir, id)):
                self.all_id2collection[id] = HiRAG_pipeline(
                    working_dir=os.path.join(self.all_collection_dir, id),
                    enable_llm_cache=config['hirag']['enable_llm_cache'],
                    embedding_func=self._embedding,
                    best_model_func=self._model,
                    cheap_model_func=self._model,
                    enable_hierachical_mode=config['hirag']['enable_hierachical_mode'],
                    embedding_batch_num=config['hirag']['embedding_batch_num'],
                    embedding_func_max_async=config['hirag']['embedding_func_max_async'],
                    enable_naive_rag=config['hirag']['enable_naive_rag']
                )

        if "persist_directory" in config:
            self.hirag = HiRAG_pipeline(
                working_dir=config['persist_directory'],
                enable_llm_cache=config['hirag']['enable_llm_cache'],
                embedding_func=self._embedding,
                best_model_func=self._model,
                cheap_model_func=self._model,
                enable_hierachical_mode=config['hirag']['enable_hierachical_mode'],
                embedding_batch_num=config['hirag']['embedding_batch_num'],
                embedding_func_max_async=config['hirag']['embedding_func_max_async'],
                enable_naive_rag=config['hirag']['enable_naive_rag']
            )
        else:
            self.hirag = HiRAG_pipeline(
                working_dir=os.path.join("./collections_HiRAG", "temp"),
                enable_llm_cache=config['hirag']['enable_llm_cache'],
                embedding_func=self._embedding,
                best_model_func=self._model,
                cheap_model_func=self._model,
                enable_hierachical_mode=config['hirag']['enable_hierachical_mode'],
                embedding_batch_num=config['hirag']['embedding_batch_num'],
                embedding_func_max_async=config['hirag']['embedding_func_max_async'],
                enable_naive_rag=config['hirag']['enable_naive_rag']
            )

        self.mode = config['hirag']['mode']
        self.top_m = config['hirag']['top_m']
        self.only_need_context = config['hirag']["only_need_context"]

    async def initialize(self):
        pass
    
    def index(self, docs):
        self.hirag.insert(docs)

    def retrieve(self, query, retrieval_top_k=5, parsed=None):
        if self.only_need_context:
            param = QueryParam(mode=self.mode, only_need_context=True,
                              top_k=retrieval_top_k, top_m=self.top_m)
        else:
            param = QueryParam(mode=self.mode, top_k=retrieval_top_k, top_m=self.top_m)
        result = self.hirag.query(query, param=param)

        if not self.only_need_context:
            return result # return the answer to the query
        else: 
            if parsed is None:
                return result  # return the raw result if no parsing is needed
            
            # recover the csv from result
            def extract_csv(section_name):
                # Regex to extract CSV between section header and triple backticks
                pattern = rf"-----{re.escape(section_name)}-----\s*```csv\n(.*?)```"
                match = re.search(pattern, result, re.DOTALL)
                if match:
                    return match.group(1)
                return None

            Backgrounds = extract_csv("Backgrounds")
            Reasoning_Path = extract_csv("Reasoning Path")
            Detail_Entity_Information = extract_csv("Detail Entity Information")
            Source_Documents = extract_csv("Source Documents")

            # Optionally, parse CSVs into pandas DataFrames:
            dfs = {k: pd.read_csv(io.StringIO(v), sep=",|\t", engine="python") if v else None
                   for k, v in {
                       "Backgrounds": Backgrounds,
                       "Reasoning_Path": Reasoning_Path,
                       "Detail_Entity_Information": Detail_Entity_Information,
                       "Source_Documents": Source_Documents
                   }.items()}

            if parsed == "csv":
                return {
                    "Backgrounds": Backgrounds,
                    "Reasoning_Path": Reasoning_Path,
                    "Detail_Entity_Information": Detail_Entity_Information,
                    "Source_Documents": Source_Documents
                }
            elif parsed == "df":
                return {
                    "Backgrounds": dfs["Backgrounds"],
                    "Reasoning_Path": dfs["Reasoning_Path"],
                    "Detail_Entity_Information": dfs["Detail_Entity_Information"],
                    "Source_Documents": dfs["Source_Documents"]
                }

    def retrieve_batch(self, queries, retrieval_top_k=5):
        return [self.retrieve(query, retrieval_top_k) for query in queries]
    
    async def rag_qa(self, question, id=None, retrieval_top_k=None):
        """
        Run RAG QA for a single question and return the final answer and the top retrieved doc (as string).
        """
        top_k = retrieval_top_k if retrieval_top_k is not None else self.config.get("retrieval_top_k", 5)
        if id:
            hirag = self.all_id2collection.get(id)
        else:
            hirag = self.hirag
        
        result = await hirag.aquery(question, param=QueryParam(mode=self.mode, top_k=top_k, top_m=self.top_m, need_context_response=True))
        if isinstance(result, tuple):
            content, response = result
        else:
            response = result
            content = "No context available"
        return response, content
    
    def check_existing_collection(self):
        return os.listdir(self.all_collection_dir)
    
    async def construct(self, id, docs, mode):
        save_dir = os.path.join(self.all_collection_dir, id)
        if mode == "new":
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            os.makedirs(save_dir, exist_ok=True)
            hirag = HiRAG_pipeline(
                working_dir=save_dir,
                embedding_func=self._embedding,
                best_model_func=self._model,
                cheap_model_func=self._model,
                enable_hierachical_mode=self.config['hirag']['enable_hierachical_mode'],
                embedding_batch_num=self.config['hirag']['embedding_batch_num'],
                embedding_func_max_async=self.config['hirag']['embedding_func_max_async'],
                enable_naive_rag=self.config['hirag']['enable_naive_rag']
            )
            self.all_id2collection[id] = hirag
        else:  # update
            if id not in self.all_id2collection:
                raise ValueError(f"Collection with id {id} does not exist.")
            hirag = self.all_id2collection[id]

        try:
            await hirag.ainsert(docs)
            # update neo4j
            stat, error = self.updateNeo4j(id)
            if error:
                logger.error(f"Error updating Neo4j for id {id}: {error}")
                return None, error
            else:
                logger.info(f"Neo4j updated successfully for id {id}")
                return NEO4J_SERVER_URI, None # no error
        except Exception as e:
            logger.error(f"Error during HiRAG construction: {str(e)}", exc_info=True)
            return None, str(e)
        
    def _parse_graphml_to_dicts(self, graphml_path):
        """
        Parse a graph_chunk_entity_relation.graphml file and return a list of dicts
        in the format expected by Neo4jLoader.add_from_dicts().
        Each dict contains 'entities' and 'relations'.
        Now supports extra attributes (e.g., description) for both nodes and edges.
        """
        tree = ET.parse(graphml_path)
        root = tree.getroot()
        ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}

        # Map GraphML keys to their attr.name
        key_map = {}
        for key in root.findall('g:key', ns):
            key_map[key.attrib['id']] = key.attrib.get('attr.name', key.attrib['id'])

        # Parse nodes (entities) with all attributes
        entities = []
        node_id_to_name = {}
        node_id_to_type = {}
        for node in root.findall('.//g:node', ns):
            node_id = node.attrib['id']
            entity_dict = {'id': node_id}  # Use node id as entity id
            for data in node.findall('g:data', ns):
                attr = key_map.get(data.attrib['key'], data.attrib['key'])
                entity_dict[attr] = data.text
            # Standardize keys for Neo4jLoader
            if 'entity_type' in entity_dict:
                entity_dict['type'] = entity_dict['entity_type']
                node_id_to_type[node_id] = entity_dict['entity_type']
            node_id_to_name[node_id] = node_id  # Always map node_id to itself
            entities.append(entity_dict)

        # Parse edges (relations) with all attributes
        relations = []
        for edge in root.findall('.//g:edge', ns):
            rel_dict = {}
            rel_dict['source'] = edge.attrib['source']
            rel_dict['target'] = edge.attrib['target']
            for data in edge.findall('g:data', ns):
                attr = key_map.get(data.attrib['key'], data.attrib['key'])
                rel_dict[attr] = data.text
            # Standardize keys for Neo4jLoader
            rel_dict['head'] = node_id_to_name.get(rel_dict['source'])
            rel_dict['tail'] = node_id_to_name.get(rel_dict['target'])
            # Prefer 'keywords' as type, fallback to 'related_to'
            rel_dict['type'] = rel_dict.get('keywords', 'related_to')
            relations.append(rel_dict)

        # Remove entities/relations without required fields
        entities = [e for e in entities if 'id' in e and 'type' in e]
        relations = [r for r in relations if r.get('head') and r.get('tail') and r.get('type')]

        return [{'entities': entities, 'relations': relations}]

    def updateNeo4j(self, id):
        """
        Parse the graphml file for the given id and update Neo4j using NEO4JLOADER.
        """
        graphml_path = os.path.join(
            self.all_collection_dir, id, "graph_chunk_entity_relation.graphml"
        )
        if not os.path.exists(graphml_path):
            logger.error(f"GraphML file not found for id {id}: {graphml_path}")
            return 1, f"GraphML file not found for id {id}: {graphml_path}"
        
        dicts = self._parse_graphml_to_dicts(graphml_path)

        with open(os.path.join(self.all_collection_dir, id, "neo4j.json"), "w", encoding="utf-8") as f: # for debug
            import json
            json.dump(dicts, f, ensure_ascii=False, indent=2)

        NEO4JLOADER.add_from_dicts(dicts, id)
        logger.info(f"Neo4j updated for KG id {id}")
        return 0, None  # no error

    def delete_KG(self, id):
        """
        Remove all nodes and relations in Neo4j for the given KG id.
        delete the folder in all_collection_dir
        """
        NEO4JLOADER.clear_kg_by_import_id(id)
        logger.info(f"Neo4j cleared for KG id {id}")

        collection_path = os.path.join(self.all_collection_dir, id)
        if os.path.exists(collection_path):
            shutil.rmtree(collection_path)
            logger.info(f"Collection {id} deleted from {self.all_collection_dir}")
        else:
            logger.warning(f"Collection {id} does not exist in {self.all_collection_dir}")
        
        if id in self.all_id2collection:
            del self.all_id2collection[id]
            logger.info(f"Removed LightRAG instance for id {id} from all_id2collection")
        else:
            logger.warning(f"No LightRAG instance found for id {id} in all_id2collection")
    
#####################################################################################################################################
# independent index parse and checking

def index_HiRAG_with_multiple_files(config):
    data_folder_path = config.get("data_dir", "./data")
    logger.info("Starting to build HiRAG with multiple files")
    all_text_data = []
    db_logger = DBLogger("HiRAG_log.json")
    changed_files = db_logger.check_files_changed(data_folder_path, file_pattern="*.jsonl")
    if not changed_files:
        logger.info("No changes in JSONL files, skipping HiRAG indexing.")
        return None, None

    logger.info(f"Found {len(changed_files)} changed files: {changed_files}")
    logger.info("Index HiRAG with changed files")

    # Collect all JSONL JSON file data
    for file in sorted(os.listdir(data_folder_path)):
        if file.endswith(".jsonl") or file.endswith(".json"):
            data_path = os.path.join(data_folder_path, file)
            logger.info(f"Loading data from {data_path}")
            file_text_data = load_data(data_path)
            all_text_data.extend(file_text_data)

    # clean up the persist directory if it exists
    persist_directory = config['persist_directory']
    if os.path.exists(persist_directory):
        logger.info(f"Persist directory {persist_directory} already exists, removing it")
        shutil.rmtree(persist_directory)
    os.makedirs(persist_directory, exist_ok=True)

    # index pharse
    HiRAG_instance = HiRAG(config)
    all_text_data =all_text_data[:100]
    HiRAG_instance.index(docs=all_text_data)

    db_logger.save_db_build_log("HiRAG", len(all_text_data))
    logger.info(f"HiRAG indexed {len(all_text_data)} documents successfully at {config['persist_directory']}.")

def check_HiRAG_index(config, test_query):
    HiRAG_instance = HiRAG(config)
    logger.info(f"Checking HiRAG index with test query: {test_query}")

    result = HiRAG_instance.retrieve(test_query)
    logger.info(f"Result: {result}")
    logger.info(f"HiRAG index check successful.")

    return True