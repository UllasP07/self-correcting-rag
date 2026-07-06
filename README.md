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

## Layout

```
src/rag/
  config.py         # env-driven settings (single source of truth)
  ollama_client.py  # raw HTTP to Ollama: embed() + chat()
  chunking.py       # recursive char splitter (M1) → semantic/parent-child (M2)
  loaders.py        # pdf / xlsx / md-txt → text
  vectorstore.py    # Qdrant: ensure_collection / upsert / search
  ingest.py         # data/ → chunks → Qdrant   (CLI: python -m src.rag.ingest)
  pipeline.py       # question → retrieve → generate  (linear, no graph yet)
  cli.py            # ask questions   (CLI: python -m src.rag.cli)
docker/docker-compose.yml   # Qdrant now; Postgres/ES/Prometheus later
data/                       # your documents
```
