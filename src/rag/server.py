"""FastAPI service exposing the RAG pipeline + Prometheus metrics (Milestone 8).

Turns the CLI into a service so a monitoring stack can watch it:

    GET  /health   → liveness check
    POST /ask      → {"question": "..."} → the full Answer (route, status, …)
    GET  /metrics  → Prometheus exposition (scraped by the Docker stack)

Run it:  python -m src.rag.server   (uvicorn on SERVER_PORT, default 8000)

Each /ask records runtime metrics (latency, route, status, retrieval score,
guardrail/HITL events); /metrics also surfaces the latest eval scores.
"""
from __future__ import annotations

import time

from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel
from starlette.responses import Response

from . import metrics
from .config import settings
from .pipeline import answer_question

app = FastAPI(title="Self-Correcting RAG", version="8.0")
metrics.register_eval_collector()


class AskRequest(BaseModel):
    question: str


class AskResponse(BaseModel):
    answer: str
    route: str
    status: str
    top_score: float = 0.0
    sql: str | None = None
    request_id: str | None = None
    guard_findings: list[str] = []


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    start = time.perf_counter()
    result = answer_question(req.question)
    metrics.record_query(result, time.perf_counter() - start)
    return AskResponse(
        answer=result.text,
        route=result.route,
        status=result.status,
        top_score=result.top_score,
        sql=result.sql,
        request_id=result.request_id,
        guard_findings=result.guard_findings,
    )


@app.get("/metrics")
def prometheus_metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def main() -> None:
    import uvicorn
    uvicorn.run(app, host=settings.server_host, port=settings.server_port)


if __name__ == "__main__":
    main()
