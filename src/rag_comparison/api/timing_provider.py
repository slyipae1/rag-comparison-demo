import os
from openai import OpenAI
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIClient:
    def __init__(self, base_url=None, api_key=None):
        self.client = OpenAI(
            api_key=api_key or os.getenv("API_KEY", "EMPTY"),
            base_url=base_url or os.getenv("BASE_URL", "http://localhost:3000/v1"),
        )
        logger.info(f"APIClient initialized with base_url: {self.client.base_url}")

    def generate_nonstream(self, messages, max_new_tokens: int = 16384):
        try:
            result = self.client.chat.completions.create(
                model="LLM",  # Model name can be adjusted as needed
                messages=messages,
                max_tokens=max_new_tokens,
                stream=False,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": True},
                    "separate_reasoning": True
                }
            )
            content = result.choices[0].message.content
            reasoning_content = result.choices[0].message.reasoning_content if hasattr(result.choices[0].message, 'reasoning_content') else ""
            logger.info(f"Reasoning contnet: {reasoning_content}")
            return {"content": content, "reasoning": reasoning_content, "inference_time": 0} # take 0 tiem to prepare
        except Exception as e:
            logger.error(f"Error in generate_nonstream: {str(e)}")
            raise
    def generate_stream(self, messages, max_new_tokens: int = 16384):
        try:
            result = self.client.chat.completions.create(
                model="LLM",  # Model name can be adjusted as needed
                messages=messages,
                max_tokens=max_new_tokens,
                stream=True,
                extra_body={
                    "chat_template_kwargs": {"enable_thinking": True},
                    "separate_reasoning": True
                }
            )
            for chunk in result:
                content = chunk.choices[0].delta.content if hasattr(chunk.choices[0].delta, 'content') else ""
                reasoning_content = chunk.choices[0].delta.reasoning_content if hasattr(chunk.choices[0].delta, 'reasoning_content') else ""
                yield {"content": content, "reasoning": reasoning_content, "inference_time": 0}  # take 0 tiem to prepare
        except Exception as e:
            logger.error(f"Error in generate_stream: {str(e)}")
            raise

# create an instance of APIClient
client = APIClient()