# Configuration reference

Every setting is read once from the environment (`.env`) into a single frozen
`Settings` object in [`src/rag/config.py`](../src/rag/config.py). Copy
`.env.example` to `.env` and override what you need; all values have sensible
local-first defaults. Tables below are grouped by the milestone that introduced
them.

## Ollama / models (M1)

| Variable | Default | Purpose |
|----------|---------|---------|
| `OLLAMA_URL` | `http://localhost:11434` | Base URL for the local Ollama server |
| `CHAT_MODEL` | `qwen2.5:7b` | Chat/generation model |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `EMBED_DIM` | `768` | Embedding dimension — **must** match `EMBED_MODEL` (nomic = 768, Azure `text-embedding-3-small` = 1536) |

## Qdrant vector DB (M1)

| Variable | Default | Purpose |
|----------|---------|---------|
| `QDRANT_URL` | `http://localhost:6333` | Qdrant REST endpoint |
| `QDRANT_COLLECTION` | `enterprise_docs` | Collection name for document vectors |

## Chunking (M1 / M2)

| Variable | Default | Purpose |
|----------|---------|---------|
| `CHUNK_SIZE` | `800` | Target characters per parent chunk |
| `CHUNK_OVERLAP` | `120` | Overlap between adjacent chunks |
| `CHILD_CHUNK_SIZE` | `350` | Smaller child chunks for precise matching (parent = full section) |

## Retrieval + reranking (M1 / M3)

| Variable | Default | Purpose |
|----------|---------|---------|
| `TOP_K` | `5` | Final chunks fed to the LLM |
| `RERANK` | `true` | Enable BGE cross-encoder reranking (first run downloads ~1 GB + PyTorch) |
| `RETRIEVE_K` | `20` | Wide candidate pool fetched before reranking |
| `RERANK_MODEL` | `BAAI/bge-reranker-base` | Cross-encoder model |

## CRAG self-correction (M4)

| Variable | Default | Purpose |
|----------|---------|---------|
| `CRAG` | `true` | Enable the grade→rewrite→retry loop (`false` = linear retrieve→generate) |
| `CRAG_GRADE_MIN_SCORE` | `0.6` | Top rerank score at/above which the LLM grader is skipped (fast pre-gate) |
| `CRAG_MAX_REWRITES` | `1` | Query rewrites attempted on weak retrieval |

## Text-to-SQL routing (M5)

| Variable | Default | Purpose |
|----------|---------|---------|
| `SQL_ENABLED` | `true` | Route data questions to text-to-SQL (`false` = everything to documents) |
| `SQLITE_PATH` | `data/acme.db` | Local structured DB (build with `python -m src.rag.seed_db`) |
| `SQL_MAX_ROWS` | `50` | Cap on rows fed back to the LLM |

## Human-in-the-loop approval (M6)

| Variable | Default | Purpose |
|----------|---------|---------|
| `HITL_ENABLED` | `true` | Freeze risky SQL for admin approval (`false` = execute all valid SQL) |
| `SENSITIVE_COLUMNS` | `salary` | Comma-separated columns that require approval to read |
| `HITL_FLAG_BROAD_SCANS` | `true` | Also freeze unbounded scans (no WHERE/LIMIT, non-aggregate) |
| `CHECKPOINT_PATH` | `data/checkpoints.db` | Durable LangGraph checkpoint + approvals registry |

## I/O guardrails (M7)

| Variable | Default | Purpose |
|----------|---------|---------|
| `GUARDRAILS_ENABLED` | `true` | Enable the input/output firewall |
| `GUARD_INJECTION` | `true` | Block prompt-injection attempts |
| `GUARD_TOXICITY` | `true` | Screen toxic input and output |
| `GUARD_USE_LLM` | `false` | Optional LLM second opinion (heuristics always run first) |
| `PII_BACKEND` | `regex` | PII masker: `regex` (zero-dep) or `presidio` (lazy, heavier) |
| `PII_ENTITIES` | `EMAIL,PHONE,SSN,CREDIT_CARD,IP` | Entity types to mask in answers |

## Eval + observability (M8)

| Variable | Default | Purpose |
|----------|---------|---------|
| `SERVER_HOST` | `0.0.0.0` | FastAPI bind host |
| `SERVER_PORT` | `8000` | FastAPI port (Prometheus scrapes `/metrics` here) |
| `EVAL_SET_PATH` | `data/eval_set.json` | Labeled questions for `python -m src.rag.eval` |
| `EVAL_REPORT_PATH` | `data/eval_report.json` | Where eval writes scores; `/metrics` surfaces them |

## Provider selection (M1.5)

Chat and embeddings pick a backend independently, so you can run cloud chat with
free local embeddings.

| Variable | Default | Purpose |
|----------|---------|---------|
| `CHAT_PROVIDER` | `ollama` | `ollama` (local) or `azure` |
| `EMBED_PROVIDER` | `ollama` | `ollama` (local) or `azure` |

### Azure OpenAI (only when a provider above is `azure`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `AZURE_OPENAI_ENDPOINT` | `` | `https://<resource>.openai.azure.com/` |
| `AZURE_OPENAI_API_KEY` | `` | API key |
| `AZURE_OPENAI_API_VERSION` | `2024-10-21` | API version |
| `AZURE_CHAT_DEPLOYMENT` | `gpt-4o-mini` | Your chat *deployment* name (not the base model) |
| `AZURE_EMBED_DEPLOYMENT` | `text-embedding-3-small` | Your embedding *deployment* name |

> **Embedder guard:** the Qdrant collection is stamped with the embedder that
> built it. Querying with a different embedder is refused (cross-embedder
> similarity is meaningless) — re-ingest with `--recreate` after switching
> `EMBED_PROVIDER`/`EMBED_MODEL`, and keep `EMBED_DIM` in sync.
