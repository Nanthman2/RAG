"""Fine wrapper around the Ollama Python client for embeddings and chat generation."""

import ollama

from config import EMBED_MODEL, LLM_MODEL


def embed(text: str) -> list[float]:
    response = ollama.embeddings(model=EMBED_MODEL, prompt=text)
    return response["embedding"]


def generate(prompt: str) -> str:
    response = ollama.chat(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1},
    )
    return response["message"]["content"]
