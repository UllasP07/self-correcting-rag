"""Tests for the FastAPI service (Milestone 8).

answer_question is monkeypatched, so /ask exercises the endpoint wiring + metrics
recording without any LLM/Qdrant/DB.
"""
import pytest

pytest.importorskip("fastapi")

from fastapi.testclient import TestClient

from src.rag import server
from src.rag.pipeline import Answer
from src.rag.vectorstore import Hit

client = TestClient(server.app)


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ask_returns_answer_fields(monkeypatch):
    fake = Answer(text="16 weeks", hits=[Hit(text="c", score=0.7, source="hb.md")],
                  top_score=0.7, route="documents", status="executed")
    monkeypatch.setattr(server, "answer_question", lambda q: fake)
    r = client.post("/ask", json={"question": "parental leave?"})
    assert r.status_code == 200
    body = r.json()
    assert body["answer"] == "16 weeks"
    assert body["route"] == "documents"
    assert body["status"] == "executed"
    assert body["top_score"] == 0.7


def test_metrics_endpoint_exposes_counters(monkeypatch):
    fake = Answer(text="ok", hits=[], top_score=0.0, route="structured",
                  status="executed", sql="SELECT 1")
    monkeypatch.setattr(server, "answer_question", lambda q: fake)
    client.post("/ask", json={"question": "how many?"})
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "rag_requests_total" in r.text
    assert "rag_request_latency_seconds" in r.text
