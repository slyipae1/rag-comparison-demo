from .base import Graph_RAG
from hipporag import HippoRAG as HippoRAG_pipeline

import os
import shutil

import xml.etree.ElementTree as ET
from rag_comparison.utils.neo4j_loader import NEO4JLOADER, NEO4J_SERVER_URI
import igraph
import os

from rag_comparison.utils.data_loading import load_data
from rag_comparison.instrumentation.logging_utils import setup_logger
from rag_comparison.utils.db_logger import DBLogger
logger = setup_logger("HippoRAG")

class HippoRAG(Graph_RAG):
    def __init__(self, config):
        """
        Initializes the HippoRAG instance with the specified parameters.
        """
        self.config = config
        self.llm_config = self.config.get("internal_LLM")

        if "persist_directory" in config:
            self.hipporag = HippoRAG_pipeline(save_dir=config.get("persist_directory"), 
                llm_model_name=self.llm_config.get("model_name", "internal_LLM"),
                llm_base_url=self.llm_config.get("base_url"),
                embedding_model_name=config.get("embedding_model")
                ) 
        else:
            self.hipporag = HippoRAG_pipeline(save_dir=os.path.join("./collections_HippoRAG", "temp"),
                llm_model_name=self.llm_config.get("model_name", "internal_LLM"),
                llm_base_url=self.llm_config.get("base_url"),
                embedding_model_name=config.get("embedding_model")
                )
            
        if "all_collection_dir" in config:
            self.all_collection_dir = config["all_collection_dir"]
        else:
            self.all_collection_dir = "./collections_HippoRAG"
        if not os.path.exists(self.all_collection_dir):
            os.makedirs(self.all_collection_dir, exist_ok=True)

        self.all_id2collection = {}
        for id in os.listdir(self.all_collection_dir):
            if os.path.isdir(os.path.join(self.all_collection_dir, id)):
                self.all_id2collection[id] = HippoRAG_pipeline(
                    save_dir=os.path.join(self.all_collection_dir, id),
                    llm_model_name=self.llm_config.get("model_name", "internal_LLM"),
                    llm_base_url=self.llm_config.get("base_url"),
                    embedding_model_name=config.get("embedding_model")
                )

    async def initialize(self):
        pass
    
    def index(self, docs):
        self.hipporag.index(docs=docs)

    def retrieve(self, query, retrieval_top_k=5):
        retrieval_results = self.hipporag.retrieve(queries=[query], num_to_retrieve=retrieval_top_k)
        # retrieval_results are of type "QuerySolution"
        docs = retrieval_results[0].docs if retrieval_results else []
        return docs

    def retrieve_batch(self, queries, retrieval_top_k=5):
        retrieval_results = self.hipporag.retrieve(queries=queries, num_to_retrieve=retrieval_top_k)
        docs_batch = [result.docs for result in retrieval_results]
        return docs_batch
    
    async def construct(self, id, docs, mode):
        save_dir = os.path.join(self.all_collection_dir, id)
        if mode == "new":
            if os.path.exists(save_dir):
                shutil.rmtree(save_dir)
            os.makedirs(save_dir, exist_ok=True)
            # create a new HippoRAG instance
            hipporag = HippoRAG_pipeline(save_dir=save_dir,
                llm_model_name=self.llm_config.get("model_name", "internal_LLM"),
                llm_base_url=self.llm_config.get("base_url"),
                embedding_model_name=self.config.get("embedding_model")
            )
            self.all_id2collection[id] = hipporag
        else:# update 
            if id not in self.all_id2collection:
                raise ValueError(f"Collection with id {id} does not exist.")
            hipporag = self.all_id2collection[id]

        try:
            hipporag.index(docs=docs) # incremental by default
            # update neo4j
            stat, error = self.updateNeo4j(id)
            if error:
                logger.error(f"Error updating Neo4j for id {id}: {error}")
                return None, error
            else:
                logger.info(f"Neo4j updated successfully for id {id}")
                return NEO4J_SERVER_URI, None # no error
        except Exception as e:
            logger.error(f"Error during HippoRAG construction: {str(e)}", exc_info=True)
            return None, str(e)

    def check_existing_collection(self):
        return os.listdir(self.all_collection_dir)
    
    async def rag_qa(self, question, id=None, retrieval_top_k=None):
        """
        Run RAG QA for a single question and return the final answer and the top retrieved doc (as string).
        """
        top_k = retrieval_top_k if retrieval_top_k is not None else self.config.get("retrieval_top_k", 5)
        if id:
            hipporag = self.all_id2collection.get(id)
        else:
            hipporag = self.hipporag

        # Call rag_qa with a single-question list
        queries_solutions, _, _ = hipporag.rag_qa([question])

        # Get the first (and only) result
        sol = queries_solutions[0]
        final_answer = sol.answer
        # Get the top-1 retrieved doc as string (or empty string if none)
        retrieved_doc = "### Retrieved Documents from HippoRAG2\n\n" + sol.docs[0] if sol.docs else "No retrieved documents found."

        return final_answer, retrieved_doc
    
    def _parse_pickle_to_dicts(self, pickle_path):
        """
        Parse a graph.pickle file (igraph) and return a list of dicts
        for Neo4jLoader.add_from_dicts(), using semantic content as node/edge names.
        Each entity keeps: hash_id, id (set to content), type.
        Each relation: head/tail are the content of the corresponding nodes.
        """
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"Pickle file not found: {pickle_path}")

        g = igraph.Graph.Read_Pickle(pickle_path)

        # Build a mapping from hash_id to content for all nodes
        hashid_to_content = {}
        hashid_to_type = {}
        for v in g.vs:
            attrs = dict(v.attributes())
            hash_id = attrs.get('hash_id') or attrs.get('name')
            content = attrs.get('content', hash_id)
            hashid_to_content[hash_id] = content
            hashid_to_type[hash_id] = attrs.get('type', attrs.get('entity_type', attrs.get('node_type', 'Node')))

        # Prepare entities: keep hash_id, id (set to content), type
        entities = []
        for v in g.vs:
            attrs = dict(v.attributes())
            hash_id = attrs.get('hash_id') or attrs.get('name')
            entity_type = "entity" if str(hash_id).startswith("entity") else ("chunk" if str(hash_id).startswith("chunk") else "unknown")
            entity = {
                'hash_id': hash_id,
                'id': attrs.get('content', hash_id),
                'name': attrs.get('content', hash_id),
                'type': entity_type
            }
            entities.append(entity)

        # Prepare relations: head/tail are the content of the corresponding nodes
        relations = []
        for e in g.es:
            edge_attrs = dict(e.attributes())
            source_idx = e.source
            target_idx = e.target
            source_vertex = g.vs[source_idx]
            target_vertex = g.vs[target_idx]
            source_hash_id = source_vertex['hash_id'] if 'hash_id' in source_vertex.attributes() else source_vertex['name']
            target_hash_id = target_vertex['hash_id'] if 'hash_id' in target_vertex.attributes() else target_vertex['name']
            rel_type = edge_attrs.get('type', 'related_to')
            rel = {
                'head': hashid_to_content.get(source_hash_id, source_hash_id),
                'tail': hashid_to_content.get(target_hash_id, target_hash_id),
                'type': rel_type,
                **edge_attrs
            }
            rel.pop('source', None)
            rel.pop('target', None)
            relations.append(rel)

        return [{'entities': entities, 'relations': relations}]
    
    def updateNeo4j(self, id):
        """
        Parse the graph.pickle file for the given id and update Neo4j using NEO4JLOADER.
        """
        inner_folder = self.llm_config.get("model_name").replace("/", "_").replace(":", "_") + "_" + self.config.get("embedding_model").replace("/", "_").replace(":", "_")

        pickle_path = os.path.join(self.all_collection_dir, id, inner_folder, "graph.pickle")
        if not os.path.exists(pickle_path):
            logger.error(f"Pickle file not found for id {id}: {pickle_path}")
            return 1, f"Pickle file not found for id {id}: {pickle_path}"

        try:
            dicts = self._parse_pickle_to_dicts(pickle_path)
            # Optionally save for debug
            with open(os.path.join(self.all_collection_dir, id, "neo4j.json"), "w", encoding="utf-8") as f:
                import json
                json.dump(dicts, f, ensure_ascii=False, indent=2)

            NEO4JLOADER.add_from_dicts(dicts, id)
            logger.info(f"Neo4j updated for KG id {id}")
            return 0, None  # no error
        except Exception as e:
            logger.error(f"Error updating Neo4j for id {id}: {e}", exc_info=True)
            return 1, str(e)

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
def index_HippoRAG_with_multiple_files(config):
    data_folder_path = config.get("data_dir", "./data")
    logger.info("Starting to build HippoRAG with multiple files")
    all_text_data = []
    db_logger = DBLogger("HippoRAG_log.json")
    changed_files = db_logger.check_files_changed(data_folder_path, file_pattern="*.jsonl")
    if not changed_files:
        logger.info("No changes in JSONL files, skipping HippoRAG indexing.")
        return None, None

    logger.info(f"Found {len(changed_files)} changed files: {changed_files}")
    logger.info("Index HippoRAG with changed files")

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

    # clean up the persist directory if it exists
    persist_directory = config.get("persist_directory")
    if os.path.exists(persist_directory):
        logger.info(f"Persist directory {persist_directory} already exists, removing it")
        shutil.rmtree(persist_directory)
    os.makedirs(persist_directory, exist_ok=True)

    # index pharse
    HippoRAG_instance = HippoRAG(config)
    HippoRAG_instance.index(docs=all_text_data)

    db_logger.save_db_build_log("HippoRAG", len(all_text_data))
    logger.info(f"HippoRAG created with {len(all_text_data)} total documents at {config.get('persist_directory', './outputs')}")
    return True

def check_HippoRAG(config, test_query):
    HippoRAG_instance = HippoRAG(config)
    logger.info(f"Checking HippoRAG  index with test query: {test_query}")

    retrieval_results = HippoRAG_instance.retrieve(query=test_query, num_to_retrieve=config.get("retrieval_top_k", 5))
    logger.info(f"Retrieved {len(retrieval_results)} documents.")
    logger.info(f"Result: {retrieval_results}")
    logger.info(f"HippoRAG index check successful.")
    
    return True
