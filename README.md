# Self-Correcting Enterprise RAG & Hybrid Text-to-SQL with HITL

A learning-oriented build of a production-grade RAG system. Runs **fully local**:
Ollama for the LLM + embeddings, Docker for infrastructure. No API keys, no cost.

## The build, in milestones

Each milestone runs before the next is added, so you understand *why* every
"senior" feature exists by feeling the limitation it fixes.

| # | Milestone | Concept |
|---|-----------|---------|
| 1 | Thin-slice RAG: PDF/MD/XLSX → chunk → embed → Qdrant → answer | base RAG |
| 2 | Semantic + parent-child chunking (Unstructured) | ingestion quality |
| 3 | BGE cross-encoder reranking | retrieve-then-rerank |
| 4 | CRAG loop in LangGraph (grade → branch → web/ES fallback) | self-correction |
| 5 | Text-to-SQL node with Pydantic schema enforcement | structured routing |
| 6 | Human-in-the-loop freeze + admin approval for risky SQL | persistent state |
| 7 | Guardrails: prompt-injection / PII masking / toxicity | I/O firewall |
| 8 | Ragas eval → Prometheus/Grafana dashboards | observability |

**We are at Milestone 1.**

## Prerequisites (installed)

- Python 3.12, Ollama (`llama3.1:8b` + `nomic-embed-text`), Docker Desktop.

## Quick start

```bash
# 0. one-time setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                       # already done

# 1. start the vector DB
docker compose -f docker/docker-compose.yml up -d
#    (make sure `ollama serve` is running too)

# 2. ingest the sample docs
python -m src.rag.ingest --recreate

# 3. ask questions
python -m src.rag.cli "How many vacation days do full-time employees get?"
python -m src.rag.cli                      # interactive mode
```

## Switching the LLM to the cloud (Azure OpenAI)

Chat and embeddings each run on a provider chosen independently, so you can keep
embeddings free/local while sending chat to the cloud. Defaults are fully local.

```bash
# .env
CHAT_PROVIDER=azure          # chat → Azure OpenAI GPT-4o mini
EMBED_PROVIDER=ollama        # embeddings stay local & free (recommended)
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_KEY=<key>
AZURE_CHAT_DEPLOYMENT=gpt-4o-mini        # your Azure *deployment* name
```

No code changes — the pipeline is unchanged; only the backend serving `chat()`
switches. (Ollama + Qdrant still need to be running for local embeddings +
vectors.)

> **Embedder guard:** each Qdrant collection is stamped with the embedder that
> built it. If you later query with a *different* embedder (e.g. switch
> `EMBED_PROVIDER` to `azure`), the app refuses rather than returning garbage —
> re-ingest with `--recreate` after changing embedders. `EMBED_DIM` must match
> the embedder (nomic = 768, Azure `text-embedding-3-small` = 1536).

## Tests

```bash
python -m pytest -q        # unit tests (chunking); no services required
```

## Layout

```
src/rag/
  config.py         # env-driven settings (single source of truth)
  llm.py            # provider facade — the ONLY place ingest/pipeline import embed()/chat()
  ollama_client.py  # local backend: raw HTTP to Ollama
  azure_client.py   # cloud backend: Azure OpenAI (opt-in via CHAT_PROVIDER=azure)
  errors.py         # friendly hints + EmbedderMismatchError guard
  chunking.py       # recursive char splitter (M1) → semantic/parent-child (M2)
  loaders.py        # pdf / xlsx / md-txt → text
  vectorstore.py    # Qdrant: ensure_collection / upsert / search / fingerprint
  ingest.py         # data/ → chunks → Qdrant   (CLI: python -m src.rag.ingest)
  pipeline.py       # question → retrieve → generate  (linear, no graph yet)
  cli.py            # ask questions   (CLI: python -m src.rag.cli)
tests/              # pytest unit tests
docker/docker-compose.yml   # Qdrant now; Postgres/ES/Prometheus later
data/                       # your documents
```
