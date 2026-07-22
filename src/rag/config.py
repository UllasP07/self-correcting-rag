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

    # --- Text-to-SQL routing (Milestone 5) ---
    # A router classifies each question: prose questions go to the CRAG document
    # loop, data questions ("how many employees in X?") go to a text-to-SQL node
    # that queries a local SQLite DB. Generated SQL is validated (read-only SELECT
    # only) before it runs.
    sql_enabled: bool = os.getenv("SQL_ENABLED", "true").lower() == "true"
    sqlite_path: str = os.getenv("SQLITE_PATH", "data/acme.db")
    sql_max_rows: int = int(os.getenv("SQL_MAX_ROWS", "50"))  # cap rows fed to the LLM

    # --- Human-in-the-loop approval for risky SQL (Milestone 6) ---
    # Risky queries FREEZE before executing: the graph checkpoints its state to
    # disk and waits for an admin to approve/deny (see src/rag/admin.py). "Risky"
    # = reads a sensitive column OR is an unbounded full-table scan.
    hitl_enabled: bool = os.getenv("HITL_ENABLED", "true").lower() == "true"
    # Comma-separated column names that require approval to read.
    sensitive_columns: tuple[str, ...] = tuple(
        c.strip().lower()
        for c in os.getenv("SENSITIVE_COLUMNS", "salary").split(",")
        if c.strip()
    )
    hitl_flag_broad_scans: bool = os.getenv("HITL_FLAG_BROAD_SCANS", "true").lower() == "true"
    # Durable LangGraph checkpoint store (persists frozen queries across restarts).
    checkpoint_path: str = os.getenv("CHECKPOINT_PATH", "data/checkpoints.db")

    # --- I/O guardrails (Milestone 7) ---
    # A firewall around answer_question: screen the INPUT for prompt-injection /
    # toxicity (block before it reaches the model), and scrub the OUTPUT (mask PII,
    # flag toxicity) before returning it.
    guardrails_enabled: bool = os.getenv("GUARDRAILS_ENABLED", "true").lower() == "true"
    guard_injection: bool = os.getenv("GUARD_INJECTION", "true").lower() == "true"
    guard_toxicity: bool = os.getenv("GUARD_TOXICITY", "true").lower() == "true"
    # Optional LLM second opinion for fuzzy injection/toxicity (heuristics run first).
    guard_use_llm: bool = os.getenv("GUARD_USE_LLM", "false").lower() == "true"
    # PII masking backend: "regex" (zero-dep default) or "presidio" (lazy, heavier).
    pii_backend: str = os.getenv("PII_BACKEND", "regex")
    # Which PII entities to mask (regex backend keys / Presidio entity names).
    pii_entities: tuple[str, ...] = tuple(
        e.strip().upper()
        for e in os.getenv("PII_ENTITIES", "EMAIL,PHONE,SSN,CREDIT_CARD,IP").split(",")
        if e.strip()
    )

    # --- Eval + observability (Milestone 8) ---
    # FastAPI service host/port (serves /ask + /metrics for Prometheus to scrape).
    server_host: str = os.getenv("SERVER_HOST", "0.0.0.0")
    server_port: int = int(os.getenv("SERVER_PORT", "8000"))
    # Local eval harness: dataset in, LLM-judged quality scores out.
    eval_set_path: str = os.getenv("EVAL_SET_PATH", "data/eval_set.json")
    eval_report_path: str = os.getenv("EVAL_REPORT_PATH", "data/eval_report.json")

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
