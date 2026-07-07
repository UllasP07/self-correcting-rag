"""Tests for the Milestone 1.5 guards: embedder fingerprint, mismatch guard,
dimension validation, and the upsert length check. All pure/fakeable — no
Qdrant or Ollama needed.
"""
from types import SimpleNamespace

import pytest

from src.rag import llm
from src.rag.config import settings
from src.rag.errors import EmbedderMismatchError
from src.rag.ingest import _check_dim
from src.rag.vectorstore import VectorStore


# --- llm.embedder_fingerprint ------------------------------------------

def test_fingerprint_ollama_default():
    assert llm.embedder_fingerprint() == {
        "provider": "ollama",
        "model": settings.embed_model,
        "dim": settings.embed_dim,
    }


def test_fingerprint_azure(monkeypatch):
    fake = SimpleNamespace(
        embed_provider="azure",
        azure_embed_deployment="text-embedding-3-small",
        embed_model="nomic-embed-text",
        embed_dim=1536,
    )
    monkeypatch.setattr(llm, "settings", fake)
    assert llm.embedder_fingerprint() == {
        "provider": "azure",
        "model": "text-embedding-3-small",
        "dim": 1536,
    }


# --- VectorStore.assert_embedder (fake read_fingerprint, no network) ----

def _vs_with_fingerprint(fp):
    vs = VectorStore.__new__(VectorStore)  # skip __init__ → no QdrantClient
    vs.read_fingerprint = lambda: fp
    return vs


def test_assert_embedder_raises_on_mismatch():
    vs = _vs_with_fingerprint(
        {"provider": "ollama", "model": "nomic-embed-text", "dim": 768}
    )
    with pytest.raises(EmbedderMismatchError):
        vs.assert_embedder(
            {"provider": "azure", "model": "text-embedding-3-small", "dim": 1536}
        )


def test_assert_embedder_passes_on_match():
    fp = {"provider": "ollama", "model": "nomic-embed-text", "dim": 768}
    _vs_with_fingerprint(fp).assert_embedder(dict(fp))  # no raise


def test_assert_embedder_noop_when_unstamped():
    # Older/unstamped collection → nothing to compare against, must not raise.
    _vs_with_fingerprint(None).assert_embedder(
        {"provider": "x", "model": "y", "dim": 1}
    )


# --- ingest._check_dim --------------------------------------------------

def test_check_dim_ok():
    _check_dim([[0.0] * settings.embed_dim])  # no raise


def test_check_dim_mismatch_raises():
    with pytest.raises(ValueError):
        _check_dim([[0.0] * (settings.embed_dim + 1)])


def test_check_dim_empty_is_noop():
    _check_dim([])  # no raise


# --- VectorStore.upsert length guard ------------------------------------

def test_upsert_length_mismatch_raises():
    vs = VectorStore.__new__(VectorStore)
    with pytest.raises(ValueError):
        vs.upsert(["a", "b"], [[0.0]], source="x.md")  # 2 texts, 1 vector
