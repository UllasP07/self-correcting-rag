"""Thin client over Ollama's HTTP API.

We deliberately use raw `requests` here instead of a framework (LangChain, etc.)
so you can see exactly what a RAG system sends to an LLM:
  - an *embedding* request turns text into a vector (for similarity search)
  - a *chat* request turns a prompt into a generated answer

Ollama exposes both on localhost:11434 once `ollama serve` is running.
"""
from __future__ import annotations

from typing import Iterable

import requests

from .config import settings


def embed(texts: Iterable[str]) -> list[list[float]]:
    """Embed one or more strings into vectors using the local embedding model.

    Returns a list of vectors (one per input string). Each vector has
    `settings.embed_dim` floats. Same-meaning texts land close together in this
    space — that closeness is what vector search exploits.
    """
    texts = list(texts)
    if not texts:
        return []

    # Ollama's /api/embed accepts a list and returns a list of embeddings.
    resp = requests.post(
        f"{settings.ollama_url}/api/embed",
        json={"model": settings.embed_model, "input": texts},
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]


def embed_one(text: str) -> list[float]:
    """Convenience wrapper: embed a single string, return a single vector."""
    return embed([text])[0]


def chat(system: str, user: str, temperature: float = 0.1) -> str:
    """Send a system+user prompt to the chat model and return the text answer.

    Low temperature (0.1) keeps answers focused and less "creative" — the right
    default for grounded RAG where we want faithfulness, not imagination.
    """
    resp = requests.post(
        f"{settings.ollama_url}/api/chat",
        json={
            "model": settings.chat_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]
