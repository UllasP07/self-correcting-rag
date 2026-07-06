"""Provider facade: the ONLY place the rest of the app imports embed()/chat() from.

Dispatches to the local (Ollama) or cloud (Azure OpenAI) implementation based on
CHAT_PROVIDER / EMBED_PROVIDER in config. Chat and embeddings are chosen
independently, so the default cheap combo — cloud chat + free local embeddings —
just works. Cloud clients are imported lazily, so nothing breaks (or needs
credentials) while you're still on the local default.
"""
from __future__ import annotations

from typing import Iterable

from . import ollama_client
from .config import settings


def _chat_fn():
    if settings.chat_provider == "azure":
        from . import azure_client
        return azure_client.chat
    return ollama_client.chat


def _embed_fn():
    if settings.embed_provider == "azure":
        from . import azure_client
        return azure_client.embed
    return ollama_client.embed


def chat(system: str, user: str, temperature: float = 0.1) -> str:
    return _chat_fn()(system, user, temperature)


def embed(texts: Iterable[str]) -> list[list[float]]:
    return _embed_fn()(texts)


def embed_one(text: str) -> list[float]:
    return embed([text])[0]
