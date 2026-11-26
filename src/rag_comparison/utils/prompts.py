"""Neutral, language-preserving prompt templates."""
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


def get_answer_prompt(external_llm):
    return (
        ChatPromptTemplate.from_messages([
            (
                "system",
                "You are a helpful assistant. Answer the user's question directly using "
                "the provided context. Integrate relevant information, keep technical "
                "terms accurate, and answer in the same language as the question.",
            ),
            (
                "user",
                "Context:\n{context}\n\nQuestion: {question}\n\n"
                "Answer in the same language as the question. If the context is insufficient, "
                "say so instead of inventing details.",
            ),
        ])
        | external_llm
        | StrOutputParser()
    )


def get_direct_answer_prompt(external_llm):
    return (
        ChatPromptTemplate.from_messages([
            ("system", "You are a helpful assistant. Answer in the same language as the question."),
            ("user", "{question}"),
        ])
        | external_llm
        | StrOutputParser()
    )
