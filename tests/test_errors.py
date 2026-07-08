"""Tests for friendly_hint — including the Qdrant-by-module case that a raw
connection error (no port/name in the message) exposed during M2."""
from src.rag.errors import friendly_hint


class _FakeQdrantError(Exception):
    pass


# Simulate qdrant_client raising with a bare "Connection refused" message.
_FakeQdrantError.__module__ = "qdrant_client.http.exceptions"


def test_hint_ollama():
    assert "ollama serve" in friendly_hint(Exception("HTTPConnectionPool port=11434"))


def test_hint_qdrant_by_message():
    assert "Qdrant" in friendly_hint(Exception("connection to 6333 refused"))


def test_hint_qdrant_by_module_only():
    # No port/name in the text — must still be caught via exception module.
    hint = friendly_hint(_FakeQdrantError("[Errno 61] Connection refused"))
    assert hint is not None and "Qdrant" in hint


def test_hint_azure():
    assert "AZURE" in friendly_hint(RuntimeError("AZURE_OPENAI_ENDPOINT not set"))


def test_hint_unknown_returns_none():
    assert friendly_hint(ValueError("some unrelated error")) is None
