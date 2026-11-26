import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class Qwen3Reranker:
    def __init__(self, model_path, device="cuda", reranker_batch_size=32):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(device).eval()
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.max_length = 8192
        self.reranker_batch_size = reranker_batch_size

        self.prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)
        self.task = 'Given a web search query, retrieve relevant passages that answer the query'

    def format_instruction(self, query, doc):
        return f"<Instruct>: {self.task}\n<Query>: {query}\n<Document>: {doc}"

    def process_inputs(self, pairs):
        inputs = self.tokenizer(
            pairs, padding=False, truncation='longest_first',
            return_attention_mask=False, max_length=self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        )
        for i, ele in enumerate(inputs['input_ids']):
            inputs['input_ids'][i] = self.prefix_tokens + ele + self.suffix_tokens
        inputs = self.tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=self.max_length)
        for key in inputs:
            inputs[key] = inputs[key].to(self.model.device)
        return inputs

    @torch.no_grad()
    def score(self, query_doc_pairs):
        # query_doc_pairs: list of (query, doc)
        scores = []
        batch_size = self.reranker_batch_size
        for i in range(0, len(query_doc_pairs), batch_size):
            batch_pairs = query_doc_pairs[i:i+batch_size]
            pairs = [self.format_instruction(q, d) for q, d in batch_pairs]
            inputs = self.process_inputs(pairs)
            batch_scores = self.model(**inputs).logits[:, -1, :]
            true_vector = batch_scores[:, self.token_true_id]
            false_vector = batch_scores[:, self.token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            batch_scores = batch_scores[:, 1].exp().tolist()
            scores.extend(batch_scores)
        return scores
    # def score(self, query_doc_pairs):
    #     # query_doc_pairs: list of (query, doc)
    #     pairs = [self.format_instruction(q, d) for q, d in query_doc_pairs]
    #     inputs = self.process_inputs(pairs)
    #     batch_scores = self.model(**inputs).logits[:, -1, :]
    #     true_vector = batch_scores[:, self.token_true_id]
    #     false_vector = batch_scores[:, self.token_false_id]
    #     batch_scores = torch.stack([false_vector, true_vector], dim=1)
    #     batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    #     scores = batch_scores[:, 1].exp().tolist()
    #     return scores
    

from langchain_community.cross_encoders.base import BaseCrossEncoder

class Qwen3CrossEncoder(BaseCrossEncoder):
    def __init__(self, qwen3_reranker):
        self.qwen3_reranker = qwen3_reranker

    def score(self, pairs):
        # pairs: list of (query, doc)
        return self.qwen3_reranker.score(pairs)