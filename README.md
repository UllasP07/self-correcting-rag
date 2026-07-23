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
| 4 | CRAG loop in LangGraph (grade → branch → query-rewrite retry) | self-correction |
| 5 | Text-to-SQL node with Pydantic schema enforcement | structured routing |
| 6 | Human-in-the-loop freeze + admin approval for risky SQL | persistent state |
| 7 | Guardrails: prompt-injection / PII masking / toxicity | I/O firewall |
| 8 | Ragas eval → Prometheus/Grafana dashboards | observability |

**All 8 milestones complete.** The pipeline is a routed, self-correcting,
guarded RAG + text-to-SQL service with human-in-the-loop approval, a local eval
harness, and Prometheus/Grafana observability.
Layout-aware PDF ingestion (Unstructured/LlamaParse) is deferred (M2B) until
there's a real table-heavy PDF to test against — current data is clean markdown.

### How M8 adds eval + observability

The pipeline now runs as a service with a monitoring stack:

```
python -m src.rag.server          # FastAPI: POST /ask, GET /metrics, GET /health (:8000)
python -m src.rag.eval            # score answer quality → data/eval_report.json
docker compose -f docker/docker-compose.yml up -d   # + Prometheus (:9090) + Grafana (:3000)
```

- **Eval** — a local Ragas-style harness scores *faithfulness*, *answer
  relevancy*, and *context precision* via an LLM judge over Ollama (no `ragas`
  dependency; judges injectable for tests).
- **Observability** — every `/ask` records runtime metrics (latency, route,
  status, retrieval score, guardrail/HITL events); `/metrics` also surfaces the
  latest eval scores (a custom collector reads the report at scrape time).
  Prometheus scrapes the host app; Grafana auto-provisions a datasource + the
  **Self-Correcting RAG** dashboard (http://localhost:3000, anonymous admin).

### How M7 firewalls I/O

`answer_question` is now wrapped by a guardrail firewall — a check on the way in
and on the way out:

```
question → [INPUT guard] → route → docs/SQL → answer → [OUTPUT guard] → answer
             │                                            │
             └─ injection/toxic → 🛡️ block (no LLM call)   └─ mask PII, flag toxicity
```

Detection is two-tier like the grader/router: deterministic heuristics first,
with an optional LLM judge (`GUARD_USE_LLM=true`) for paraphrased attacks. A
blocked input short-circuits — no embedding, retrieval, SQL, or model call. PII
masking defaults to a zero-dependency regex backend (email/phone/SSN/card/IP);
Presidio is optional and lazy-loaded only when `PII_BACKEND=presidio`. Set
`GUARDRAILS_ENABLED=false` to disable the firewall.

### How M6 gates risky SQL (persistent HITL)

A generated query that reads a **sensitive column** (e.g. `salary`) or is an
**unbounded scan** doesn't run automatically — it *freezes*. The graph
checkpoints its full state to disk (LangGraph `SqliteSaver`) and waits for a
human:

```
text-to-SQL → assess risk ─┬─ safe  → execute → answer
                           └─ risky → 🧊 freeze (checkpoint) ─ approve? ─┬─ yes → execute → answer
                                                                         └─ no  → denied
```

Because state is durable, a frozen query survives a process restart — approve it
later from a separate terminal:

```bash
python -m src.rag.admin list                 # what's waiting, the SQL, why flagged
python -m src.rag.admin approve <request_id>  # resume from the checkpoint + execute
python -m src.rag.admin deny <request_id>     # refuse
```

The CLI also offers a quick inline approve when a query freezes. Set
`HITL_ENABLED=false` to execute all valid SQL immediately (M5 behavior).

### How M5 routes (documents vs. data)

Not every question lives in prose. "What's the parental leave policy?" belongs to
the document loop; "how many engineers, and what's the average salary?" belongs
to a **database**. A router picks the pipe:

```
question → route ─┬─ documents  → CRAG loop (M1–M4)
                  └─ structured → text-to-SQL → validate → execute → answer
```

The generated SQL is never run raw: the LLM emits into a Pydantic `SQLQuery` that
enforces a **single read-only SELECT** (no INSERT/UPDATE/DELETE/DROP, no stacked
statements), and the SQLite connection is opened read-only as a second guard.
Structured data lives in a local `data/acme.db` (`python -m src.rag.seed_db`).
Set `SQL_ENABLED=false` to route everything to documents.

### How M4 self-corrects

The query path is no longer a straight line — it's a LangGraph state machine
that grades its own retrieval and retries when it's weak (fully local; the
"correction" is a query rewrite, not a web call):

```
retrieve → grade ─┬─ relevant ───────────────────→ generate
                  ├─ weak & retries left → rewrite_query → (retrieve)
                  └─ weak & out of retries ────────→ generate  (answer honestly)
```

Grading is two-tier: a fast score pre-gate (`CRAG_GRADE_MIN_SCORE`), then an LLM
judge for the murky middle. The CLI prints the correction trace so you can see
when it grades a retrieval WEAK and rewrites the query. Set `CRAG=false` to fall
back to the pre-M4 linear path.

> ⚠️ **First-run note:** with `RERANK=true` (the default), the first query
> downloads the BGE reranker model (~1 GB) and loads PyTorch. Set `RERANK=false`
> in `.env` to skip reranking (pure cosine retrieval) if you want a lighter run.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — system architecture, request
  lifecycle, and the LangGraph state machines, with diagrams.
- [docs/CONFIGURATION.md](docs/CONFIGURATION.md) — every environment variable,
  grouped by milestone, with defaults and what it does.

## Prerequisites (installed)

- Python 3.12, Ollama (`qwen2.5:7b` chat + `nomic-embed-text` embeddings), Docker Desktop.

## Quick start

```bash
# 0. one-time setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env                       # already done

# 1. start the vector DB
docker compose -f docker/docker-compose.yml up -d
#    (make sure `ollama serve` is running too)

# 2. ingest the sample docs  +  seed the sample database (M5)
python -m src.rag.ingest --recreate
python -m src.rag.seed_db                  # builds data/acme.db (employees table)

# 3. ask questions — routed automatically to docs or SQL
python -m src.rag.cli "How many vacation days do full-time employees get?"   # → documents
python -m src.rag.cli "What is the average salary per department?"           # → SQL
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
python -m pytest -q        # unit tests (chunking, errors, guards, loaders, pipeline, reranker, grader, graph, router, text-to-SQL, database, risk, HITL freeze/resume, guardrails, eval, metrics, server); no Ollama/Qdrant/DB services needed
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
  reranker.py       # BGE cross-encoder reranking (M3)
  ingest.py         # data/ → chunks → Qdrant   (CLI: python -m src.rag.ingest)
  grader.py         # CRAG retrieval grading — score pre-gate + LLM judge (M4)
  router.py         # documents-vs-structured routing (M5)
  text_to_sql.py    # LLM → validated read-only SQLQuery (Pydantic) (M5)
  database.py       # SQLite: schema_ddl / read-only run_select (M5)
  seed_db.py        # build data/acme.db employees table  (CLI: python -m src.rag.seed_db)
  risk.py           # SQL risk policy: sensitive columns + broad scans (M6)
  approvals.py      # persistent registry of frozen queries awaiting approval (M6)
  admin.py          # approve/deny frozen SQL   (CLI: python -m src.rag.admin)
  guardrails.py     # I/O firewall: prompt-injection / PII masking / toxicity (M7)
  graph.py          # LangGraph: CRAG loop (M4) + text-to-SQL + HITL gate (M5/M6)
  pipeline.py       # answer_question = input guard → core → output guard
  eval.py           # local Ragas-style quality eval  (CLI: python -m src.rag.eval)
  metrics.py        # Prometheus runtime metrics + eval-score collector (M8)
  server.py         # FastAPI /ask /metrics /health   (CLI: python -m src.rag.server)
  cli.py            # ask questions   (CLI: python -m src.rag.cli)
tests/              # pytest unit tests
docker/
  docker-compose.yml        # Qdrant + Prometheus + Grafana
  prometheus.yml            # scrape config → host app :8000
  grafana/provisioning/     # auto datasource + RAG dashboard
data/                       # documents + acme.db + checkpoints.db + eval_report.json
```
