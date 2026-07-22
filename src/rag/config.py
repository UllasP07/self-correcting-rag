"""Central configuration, loaded once from environment (.env).

Every other module imports `settings` from here so there is a single source of
truth for model names, URLs, and tuning knobs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file: src/rag/config.py).
load_dotenv()


@dataclass(frozen=True)
class Settings:
    # Ollama
    ollama_url: str = os.getenv("OLLAMA_URL", "http://localhost:11434")
    chat_model: str = os.getenv("CHAT_MODEL", "qwen2.5:7b")
    embed_model: str = os.getenv("EMBED_MODEL", "nomic-embed-text")
    embed_dim: int = int(os.getenv("EMBED_DIM", "768"))

    # Qdrant
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_collection: str = os.getenv("QDRANT_COLLECTION", "enterprise_docs")

    # Chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "120"))
    # M2 parent-child: children are smaller (precise matching); parent = full section.
    child_chunk_size: int = int(os.getenv("CHILD_CHUNK_SIZE", "350"))

    # Retrieval
    top_k: int = int(os.getenv("TOP_K", "5"))          # final chunks used to answer

    # --- Reranking (Milestone 3) ---
    # Fetch a wider candidate pool from Qdrant, then a cross-encoder re-scores
    # (query, chunk) pairs and we keep the best TOP_K.
    rerank_enabled: bool = os.getenv("RERANK", "true").lower() == "true"
    retrieve_k: int = int(os.getenv("RETRIEVE_K", "20"))   # candidates before rerank
    rerank_model: str = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-base")

    # --- CRAG self-correcting loop (Milestone 4) ---
    # After retrieve+rerank we GRADE the context. If it's too weak, we rewrite the
    # query and retrieve again (up to `crag_max_rewrites` times) before answering.
    crag_enabled: bool = os.getenv("CRAG", "true").lower() == "true"
    # Fast pre-gate: if the top (rerank) score clears this, skip the LLM grader.
    crag_grade_min_score: float = float(os.getenv("CRAG_GRADE_MIN_SCORE", "0.6"))
    crag_max_rewrites: int = int(os.getenv("CRAG_MAX_REWRITES", "1"))

    # --- Provider selection ---
    # Which backend serves chat / embeddings: "ollama" (local, free, default)
    # or "azure" (Azure OpenAI). Chosen independently so you can run cloud chat
    # + local embeddings (the cheapest sensible combo).
    chat_provider: str = os.getenv("CHAT_PROVIDER", "ollama")
    embed_provider: str = os.getenv("EMBED_PROVIDER", "ollama")

    # --- Azure OpenAI (only used when a provider above is "azure") ---
    # NOTE: on Azure, the "model" you call is your *deployment name*, which you
    # choose when you deploy a model in the Azure OpenAI / Foundry portal.
    azure_openai_endpoint: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_openai_api_key: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_openai_api_version: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
    azure_chat_deployment: str = os.getenv("AZURE_CHAT_DEPLOYMENT", "gpt-4o-mini")
    azure_embed_deployment: str = os.getenv("AZURE_EMBED_DEPLOYMENT", "text-embedding-3-small")


settings = Settings()
