"""Lightweight side-by-side comparison UI.

It calls the OpenAI-style endpoint exposed by :mod:`rag_comparison.api.app` and
shows answers, optional reasoning traces and measured stage time for each
configured pipeline.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr
import requests

CONFIG_FILE = os.getenv("RAG_CONFIG_FILE", "configs/mock-demo.json")


def load_endpoints(path: str = CONFIG_FILE) -> dict[str, str]:
    with open(path, "r", encoding="utf-8") as f:
        configs = json.load(f)
    return {
        name: f"http://{cfg.get('API_host', '127.0.0.1')}:{cfg.get('API_port', 8000)}/v1/chat/completions"
        for name, cfg in configs.items()
    }


ENDPOINTS = load_endpoints()


def query(endpoint: str, question: str):
    if not question.strip():
        return "", "", ""
    try:
        response = requests.post(
            endpoint,
            json={"messages": [{"role": "user", "content": question}], "stream": False},
            timeout=60,
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]
        content = choice["message"]["content"]
        reasoning = choice.get("reasoning_process") or ""
        timing = choice.get("inference_time")
        timing_text = f"{timing:.4f} s" if isinstance(timing, (int, float)) else "n/a"
        return content, reasoning, timing_text
    except Exception as exc:  # UI must stay up when one backend is unavailable
        return f"Request failed: {exc}", "", "n/a"


def make_demo():
    names = list(ENDPOINTS)
    with gr.Blocks(title="RAG Comparison Demo") as demo:
        gr.Markdown(
            "# RAG Comparison Demo\n"
            "Ask one question and compare answers from configured pipelines. "
            "Run the API endpoints first (see README)."
        )
        question = gr.Textbox(label="Question", lines=2, placeholder="Ask something...")
        ask = gr.Button("Compare", variant="primary")

        outputs = []
        with gr.Row():
            for name in names:
                with gr.Column():
                    gr.Markdown(f"### {name}")
                    answer = gr.Textbox(label="Answer", lines=8)
                    reasoning = gr.Textbox(label="Reasoning / retrieved context", lines=6)
                    timing = gr.Textbox(label="Time", lines=1)
                    outputs.append((answer, reasoning, timing))

        def compare(text):
            return [item for name in names for item in query(ENDPOINTS[name], text)]

        ask.click(compare, inputs=[question], outputs=[c for group in outputs for c in group])
    return demo


if __name__ == "__main__":
    make_demo().launch(server_name="127.0.0.1", server_port=7860, share=False)
