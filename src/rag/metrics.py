"""Prometheus metrics for the RAG service (Milestone 8).

Two kinds of signal end up in Prometheus:

  RUNTIME  — recorded live around each request: how many, how fast, which route,
             how good the retrieval, how often guardrails/HITL fired.
  QUALITY  — the eval harness (eval.py) writes a JSON report; a custom collector
             reads it at scrape time so answer-quality scores (faithfulness, …)
             show up on the same dashboard as live traffic — no pushgateway.

`record_query` maps an `Answer` onto the runtime metrics; the server calls it.
"""
from __future__ import annotations

import json
from pathlib import Path

from prometheus_client import REGISTRY, Counter, Histogram
from prometheus_client.core import GaugeMetricFamily

from .config import settings

_ROOT = Path(__file__).resolve().parents[2]

# --- runtime metrics -------------------------------------------------------

REQUESTS = Counter(
    "rag_requests_total", "RAG requests processed", ["route", "status"])
LATENCY = Histogram(
    "rag_request_latency_seconds", "End-to-end answer latency (seconds)")
TOP_SCORE = Histogram(
    "rag_retrieval_top_score", "Top retrieval score on the document route",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0))
GUARDRAIL_BLOCKS = Counter(
    "rag_guardrail_blocks_total", "Inputs blocked by the guardrail firewall")
PII_MASKED = Counter(
    "rag_pii_masked_total", "Answers that had PII masked on the way out")
HITL_FREEZES = Counter(
    "rag_hitl_freezes_total", "SQL queries frozen for admin approval")


def record_query(answer, latency: float) -> None:
    """Update runtime metrics from a finished `Answer`."""
    REQUESTS.labels(route=answer.route, status=answer.status).inc()
    LATENCY.observe(latency)
    if answer.route == "documents" and answer.status == "executed":
        TOP_SCORE.observe(answer.top_score)
    if answer.status == "blocked":
        GUARDRAIL_BLOCKS.inc()
    if answer.pii_masked:
        PII_MASKED.inc()
    if answer.status == "frozen":
        HITL_FREEZES.inc()


# --- quality metrics (read from the eval report at scrape time) -----------

class EvalReportCollector:
    """A custom collector that surfaces the latest eval scores as gauges."""

    def collect(self):
        g = GaugeMetricFamily(
            "rag_eval_score", "Latest RAG eval scores (0-1)", labels=["metric"])
        raw = Path(settings.eval_report_path)
        path = raw if raw.is_absolute() else _ROOT / raw
        if path.exists():
            try:
                data = json.loads(path.read_text())
            except (ValueError, OSError):
                data = {}
            for metric, value in (data.get("scores") or {}).items():
                g.add_metric([metric], float(value))
        yield g


_eval_collector_registered = False


def register_eval_collector() -> None:
    """Register the eval collector once (idempotent)."""
    global _eval_collector_registered
    if not _eval_collector_registered:
        REGISTRY.register(EvalReportCollector())
        _eval_collector_registered = True
