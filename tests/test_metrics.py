"""Tests that record_query maps an Answer onto the Prometheus counters."""
from src.rag import metrics
from src.rag.pipeline import Answer
from src.rag.vectorstore import Hit


def _val(counter, **labels):
    return counter.labels(**labels)._value.get() if labels else counter._value.get()


def test_records_request_and_latency():
    before = _val(metrics.REQUESTS, route="documents", status="executed")
    ans = Answer(text="a", hits=[Hit(text="c", score=0.8, source="d")],
                 top_score=0.8, route="documents", status="executed")
    metrics.record_query(ans, 0.42)
    after = _val(metrics.REQUESTS, route="documents", status="executed")
    assert after == before + 1


def test_blocked_increments_guardrail_counter():
    before = _val(metrics.GUARDRAIL_BLOCKS)
    metrics.record_query(
        Answer(text="", hits=[], top_score=0.0, status="blocked",
               guard_findings=["prompt-injection pattern detected"]),
        0.01)
    assert _val(metrics.GUARDRAIL_BLOCKS) == before + 1


def test_frozen_increments_freeze_counter():
    before = _val(metrics.HITL_FREEZES)
    metrics.record_query(
        Answer(text="", hits=[], top_score=0.0, route="structured", status="frozen"),
        0.01)
    assert _val(metrics.HITL_FREEZES) == before + 1


def test_pii_masked_counter():
    before = _val(metrics.PII_MASKED)
    metrics.record_query(
        Answer(text="[EMAIL]", hits=[], top_score=0.0, status="executed",
               pii_masked=True),
        0.01)
    assert _val(metrics.PII_MASKED) == before + 1
