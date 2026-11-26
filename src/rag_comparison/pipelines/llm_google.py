from .base import RAGpipeline

from rag_comparison.instrumentation.logging_utils import setup_logger
from rag_comparison.instrumentation.timing import RecordTime
logger = setup_logger("LLM_Google")

import importlib

from rag_comparison.model_utils.setup import setup_LLM

from rag_comparison.retrievers.google import GoogleRetriever

from rag_comparison.utils.prompts import get_answer_prompt, get_direct_answer_prompt

class LLM_Google(RAGpipeline):

    def __init__(self, settings_config):
        self.config = settings_config
        self.logger = logger
        self.status_messages = []
        self.timeRecord = {"PIPELINE":"LLM_Google", "TOTAL": 0, "QUESTION": "", "LLM_ANSWER": 0} # Google_Retrieval, Google_Summarization

        # setup google retriever
        self.google_retriever = GoogleRetriever(self.config)

        # setu LLMs
        self.internal_LLM = setup_LLM(self.config["internal_LLM"])
        self.internal_LLM_config = self.config["internal_LLM"]
        self.external_LLM = setup_LLM(self.config["external_LLM"])

        # setup prompts
        self.answer_prompt = get_answer_prompt(self.external_LLM)
        self.direct_answer_prompt = get_direct_answer_prompt(self.external_LLM)

        self.logger.info("Pipeline is ready.")

    
    async def process_question(self, question: str, use_stream: bool = False): 
        # reset time record
        self.timeRecord = {"PIPELINE":"LLM_Google", "QUESTION": question} # Google_Retrieval, Google_Summarization, LLM_ANSWER

        self.logger.info(f"Processing question: {question} (Stream mode: {use_stream})")
        
        import time
        inference_start = time.time()
        temp_start = time.time()

        try:
            # google retrieval
            google_raw_retrieval = self.google_retriever.invoke(question)
            self.timeRecord["Google_Retrieval"] = time.time() - temp_start
            self.logger.info(f"Google retrieval")
            self.logger.info(f"used time: {time.time() - temp_start} seconds")
            temp_start = time.time()
            google_context = self.google_retriever.summarize_google_results(google_raw_retrieval, self.internal_LLM)
            self.timeRecord["Google_Summarization"] = time.time() - temp_start
            self.logger.info(f"summerize Google retrieval")
            self.logger.info(f"used time: {time.time() - temp_start} seconds")
            temp_start = time.time()

            if not use_stream:

                if google_context is None or google_context == "":
                    prompt_data = {"question": question, "context": ""}
                    inference_end = time.time()
                    self.timeRecord["TOTAL"] = inference_end - inference_start
                    response = self.direct_answer_prompt.invoke(prompt_data)
                    self.logger.info("direct_answer_prompt invoked")
                    self.logger.info(f"used time: {time.time() - temp_start} seconds")
                    self.timeRecord["LLM_ANSWER"] = time.time() - temp_start
                    content = response.content if hasattr(response, 'content') else str(response)
                    yield {"content": content, "inference_time": inference_end - inference_start, "reasoning": "No content found."}
                    RecordTime(self.timeRecord)
                    return

                prompt_data = {"question": question, "context": google_context}
                inference_end = time.time()
                self.timeRecord["TOTAL"] = inference_end - inference_start
                response = self.answer_prompt.invoke(prompt_data)
                self.logger.info("answer_prompt invoked")
                self.logger.info(f"used time: {time.time() - temp_start} seconds")
                self.timeRecord["LLM_ANSWER"] = time.time() - temp_start
                # Check the response type and handle it accordingly
                content = response.content if hasattr(response, 'content') else str(response)
                yield {"content": content, "inference_time": inference_end - inference_start,
                        "reasoning": f"## Google Retrieval  \n{google_context}.\n\n"}
                RecordTime(self.timeRecord)
                return
                    
                
            else: # Streaming processing uses asynchronous methods
                    
                    full_response = ""
                    if google_context is None or google_context == "":
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
                        self.logger.info("direct_answer_prompt invoked")
                        self.logger.info(f"used time: {time.time() - temp_start} seconds")
                        self.timeRecord["LLM_ANSWER"] = time.time() - temp_start
                        full_response = ''.join(full_response).replace("'", "").replace(",", "").strip()
                        logger.info(f"Full response: {full_response}")
                        RecordTime(self.timeRecord)
                        return
                    
                    prompt_data = {"question": question, "context": google_context}
                    yield {"reasoning": f"## Google Retrieval  \n{google_context}.\n\n"}
                    
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
                    self.logger.info("answer_prompt invoked")
                    self.logger.info(f"used time: {time.time() - temp_start} seconds")
                    self.timeRecord["LLM_ANSWER"] = time.time() - temp_start
                            
                    full_response = ''.join(full_response).replace("'", "").replace(",", "").strip()
                    logger.info(f"Full response: {full_response}")
                    RecordTime(self.timeRecord)
                    return
                            
        except Exception as e:
            self.logger.error(f"Error in process_question: {str(e)}", exc_info=True)
            raise