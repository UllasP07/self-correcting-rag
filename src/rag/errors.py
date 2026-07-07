"""Friendly, actionable error messages for the common ways this app breaks.

Two things live here:
  - EmbedderMismatchError: raised when you query an index with a *different*
    embedder than the one that built it (the silent-wrong-answer bug the
    Milestone 1.5 guard exists to prevent).
  - friendly_hint(): turns a raw connection/config exception into a one-line
    "here's how to fix it" message, so the CLI shows guidance instead of a
    stack trace when Ollama / Qdrant / Azure aren't ready.
"""
from __future__ import annotations


class EmbedderMismatchError(RuntimeError):
    """Query embedder differs from the one that built the collection."""


def friendly_hint(exc: Exception) -> str | None:
    """Return a human fix-it hint for a recognized failure, else None."""
    msg = str(exc).lower()

    # Ollama not running (requests raises with the host:port in the message).
    if "11434" in msg:
        return (
            "Can't reach Ollama at localhost:11434.\n"
            "Start it with:  ollama serve"
        )

    # Qdrant not reachable.
    if "6333" in msg or "qdrant" in msg:
        return (
            "Can't reach Qdrant at localhost:6333.\n"
            "Start it with:  docker compose -f docker/docker-compose.yml up -d"
        )

    # Azure selected but not configured.
    if "azure_openai" in msg or ("azure" in msg and "endpoint" in msg):
        return (
            "Azure provider selected but not configured.\n"
            "Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env, "
            "or set CHAT_PROVIDER=ollama to run locally."
        )
    return None
