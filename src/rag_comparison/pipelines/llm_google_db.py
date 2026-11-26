from .base import RAGpipeline

from rag_comparison.instrumentation.logging_utils import setup_logger
from rag_comparison.instrumentation.timing import RecordTime
logger = setup_logger("LLM_Google_DB")

import importlib
import time
import asyncio

from rag_comparison.model_utils.setup import setup_LLM

from rag_comparison.retrievers.google import GoogleRetriever

from rag_comparison.utils.prompts import get_answer_prompt, get_direct_answer_prompt

class LLM_Google_DB(RAGpipeline):

    def __init__(self, settings_config):
        self.config = settings_config
        self.logger = logger
        self.status_messages = []
        self.timeRecord = {"PIPELINE":"LLM_Google_DB", "DB": self.config["DB_type"], "Retriever": self.config["Retriever_type"], "retrieval_top_k": self.config["retrieval_top_k"], "rerank_top_k": self.config["rerank_top_k"],
                            "TOTAL": 0, "QUESTION": "", "LLM_ANSWER": 0} # DB_Retrieval, Google_Retrieval, Google_Summarization, LLM_ANSWER

        # setup google retriever
        self.google_retriever = GoogleRetriever(self.config)

        # setu LLMs
        self.internal_LLM = setup_LLM(self.config["internal_LLM"])
        self.internal_LLM_config = self.config["internal_LLM"]
        self.external_LLM = setup_LLM(self.config["external_LLM"])

        # setup prompts
        self.answer_prompt = get_answer_prompt(self.external_LLM)
        self.direct_answer_prompt = get_direct_answer_prompt(self.external_LLM)

        try:
            # setup db
            DB_type = self.config["DB_type"]
            db_modules = {"MilvusDB": "milvus", "ChromaDB": "chroma", "pgvectorDB": "pgvector"}
            module = importlib.import_module(
                f'rag_comparison.backends.vector.{db_modules[DB_type]}'
            )
            DB_class = getattr(module, DB_type)
            self.DB = DB_class(self.config)

            # setup retriever
            Retriever_type = self.config["Retriever_type"]
            retriever_modules = {"CompressionRetriever": "compression", "GoogleRetriever": "google"}
            module = importlib.import_module(
                f'rag_comparison.retrievers.{retriever_modules[Retriever_type]}'
            )
            retriever_class = getattr(module, Retriever_type)
            self.retriever = retriever_class(self.DB, self.config)

            self.logger.info("Pipeline is ready.")
        except Exception as e:
            self.logger.error(f"Error during pipeline setup: {str(e)}", exc_info=True)
            raise

    def google_task(self, question):
        temp_start = time.time()
        google_raw_retrieval = self.google_retriever.invoke(question)
        self.timeRecord["Google_Retrieval"] = time.time() - temp_start
        self.logger.info(f"Google retrieval -- time: {time.time() - temp_start} seconds")
        temp_start = time.time()
        google_context = self.google_retriever.summarize_google_results(google_raw_retrieval, self.internal_LLM)
        self.logger.info(f"Summerize Google retrieval -- time: {time.time() - temp_start} seconds")
        self.timeRecord["Google_Summarization"] = time.time() - temp_start
        return google_context

    def db_task(self, question):
        temp_start = time.time()
        DB_raw_retrieval = self.retriever.invoke(question)
        DB_context = "\n\n".join([doc.page_content for doc in DB_raw_retrieval])
        self.timeRecord["DB_Retrieval"] = time.time() - temp_start
        self.logger.info(f"DB retrieval -- time: {time.time() - temp_start} seconds")
        return DB_context
    
    async def process_question(self, question: str, use_stream: bool = False): 
        # reset time record
        self.timeRecord = {"PIPELINE":"LLM_Google_DB", "DB": self.config["DB_type"], "Retriever": self.config["Retriever_type"], "retrieval_top_k": self.config["retrieval_top_k"], "rerank_top_k": self.config["rerank_top_k"],
                            "QUESTION": question} # DB_Retrieval, Google_Retrieval, Google_Summarization, LLM_ANSWER

        self.logger.info(f"Processing question: {question} (Stream mode: {use_stream})")
        
        import time
        inference_start = time.time()

        async def run_in_executor(func, *args):
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, func, *args)

        try:

            # Streaming: run both retrievals concurrently and yield as soon as each is ready
            db_task = asyncio.create_task(run_in_executor(self.db_task, question))
            google_task = asyncio.create_task(run_in_executor(self.google_task, question))

            retrievals = {"DB": None, "GOOGLE": None}
            finished = set()

            while len(finished) < 2:
                done, _ = await asyncio.wait(
                    [db_task, google_task], return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    if task is db_task and "DB" not in finished:
                        DB_context = task.result()
                        retrievals["DB"] = DB_context
                        finished.add("DB")
                        if DB_context and use_stream:
                            yield {"reasoning": f"## Vector DB & BM25 Retrieval  \n{DB_context}.\n\n"}
                    if task is google_task and "GOOGLE" not in finished:
                        google_context = task.result()
                        retrievals["GOOGLE"] = google_context
                        finished.add("GOOGLE")
                        if google_context and use_stream:
                            yield {"reasoning": f"## Google Retrieval \n{google_context}.\n\n"}

            # After both retrievals are done, build the prompt context
            retrieval_context = ""
            if retrievals["DB"]:
                retrieval_context += f"## Vector DB & BM25 Retrieval  \n{retrievals['DB']}.\n\n"
            if retrievals["GOOGLE"]:
                retrieval_context += f"## Google Retrieval \n{retrievals['GOOGLE']}.\n\n"

            temp_start = time.time()
            if not use_stream:
                if retrieval_context is None or retrieval_context == "":
                    prompt_data = {"question": question, "context": ""}
                    inference_end = time.time()
                    self.timeRecord["TOTAL"] = inference_end - inference_start
                    response = self.direct_answer_prompt.invoke(prompt_data)
                    self.timeRecord["LLM_ANSWER"] = time.time() - temp_start
                    self.logger.info("direct_answer_prompt invoked")
                    self.logger.info(f"used time: {time.time() - temp_start} seconds")
                    temp_start = time.time()
                    content = response.content if hasattr(response, 'content') else str(response)
                    yield {"content": content, "inference_time": inference_end - inference_start, "reasoning": "No content found."}
                    RecordTime(self.timeRecord)
                    return

                prompt_data = {"question": question, "context": retrieval_context}
                inference_end = time.time()
                self.timeRecord["TOTAL"] = inference_end - inference_start
                response = self.answer_prompt.invoke(prompt_data)
                self.timeRecord["LLM_ANSWER"] = time.time() - temp_start
                self.logger.info("answer_prompt invoked")
                self.logger.info(f"used time: {time.time() - temp_start} seconds")
                temp_start = time.time()
                # Check the response type and handle it accordingly
                content = response.content if hasattr(response, 'content') else str(response)
                yield {"content": content, "inference_time": inference_end - inference_start,
                        "reasoning": retrieval_context}
                RecordTime(self.timeRecord)
                return
                    
                
            else: # Streaming processing uses asynchronous methods
                full_response = ""
                if retrieval_context is None or retrieval_context == "":
                    prompt_data = {"question": question, "context": ""}
                    yield {"reasoning": "No content found."}
                    inference_end = time.time()
                    self.timeRecord["TOTAL"] = inference_end - inference_start
                    yield {"inference_time": inference_end - inference_start}

                    async for chunk in self.direct_answer_prompt.astream(prompt_data):
                        if hasattr(chunk, 'content'):
                            yield {"content": chunk.content}
                            full_response += chunk.content
                        else:
                            yield {"content": str(chunk)}
                            full_response += str(chunk)

                    self.timeRecord["LLM_ANSWER"] = time.time() - temp_start
                    self.logger.info("direct_answer_prompt invoked")
                    self.logger.info(f"used time: {time.time() - temp_start} seconds")

                    full_response = ''.join(full_response).replace("'", "").replace(",", "").strip()
                    logger.info(f"Full response: {full_response}")
                    RecordTime(self.timeRecord)
                    return
                
                prompt_data = {"question": question, "context": retrieval_context}
                
                inference_end = time.time()
                self.timeRecord["TOTAL"] = inference_end - inference_start
                yield {"inference_time": inference_end - inference_start}

                async for chunk in self.answer_prompt.astream(prompt_data):
                    if hasattr(chunk, 'content'):
                        yield {"content": chunk.content}
                        full_response += chunk.content
                    else:
                        yield {"content": str(chunk)}
                        full_response += str(chunk)
                self.timeRecord["LLM_ANSWER"] = time.time() - temp_start
                self.logger.info("answer_prompt invoked")
                self.logger.info(f"used time: {time.time() - temp_start} seconds")

                        
                full_response = ''.join(full_response).replace("'", "").replace(",", "").strip()
                logger.info(f"Full response: {full_response}")
                RecordTime(self.timeRecord)
                return
                            
        except Exception as e:
            self.logger.error(f"Error in process_question: {str(e)}", exc_info=True)
            raise