"""Tests for the CRAG graph's branching, with every step injected.

No Ollama, no Qdrant, no model downloads — we drive the state machine with fake
retrieve/grade/rewrite/generate functions and assert the control flow.
"""
import pytest

pytest.importorskip("langgraph")  # skip cleanly if the M4 dep isn't installed

from src.rag.grader import Grade
from src.rag.graph import run_crag
from src.rag.vectorstore import Hit


def _hits(tag):
    return [Hit(text=tag, score=0.5, source="x")]


def test_relevant_first_pass_no_rewrite():
    out = run_crag(
        "q",
        retrieve_fn=lambda q: _hits("first"),
        grade_fn=lambda question, hits, ctx: Grade(True, "good", False),
        rewrite_fn=lambda question, hits: pytest.fail("should not rewrite"),
        generate_fn=lambda question, hits: f"answered from {hits[0].text}",
    )
    assert out.text == "answered from first"
    assert out.correction.attempts == 1
    assert out.correction.graded_relevant is True
    assert out.correction.rewritten_query is None


def test_weak_then_rewrite_then_relevant():
    # Grade "no" on the original query, "yes" once the query has been rewritten.
    def grade_fn(question, hits, ctx):
        return Grade(hits[0].text == "rewritten", "graded", True)

    out = run_crag(
        "original question",
        retrieve_fn=lambda q: _hits("rewritten" if q == "better query" else "orig"),
        grade_fn=grade_fn,
        rewrite_fn=lambda question, hits: "better query",
        generate_fn=lambda question, hits: f"answered from {hits[0].text}",
    )
    assert out.text == "answered from rewritten"
    assert out.correction.attempts == 2
    assert out.correction.graded_relevant is True
    assert out.correction.rewritten_query == "better query"


def test_persistently_weak_gives_up_and_answers():
    # Never relevant: after exhausting rewrites, it should still generate.
    out = run_crag(
        "q",
        retrieve_fn=lambda q: _hits("weak"),
        grade_fn=lambda question, hits, ctx: Grade(False, "still weak", True),
        rewrite_fn=lambda question, hits: "rewritten q",
        generate_fn=lambda question, hits: "honest best-effort answer",
    )
    assert out.text == "honest best-effort answer"
    assert out.correction.graded_relevant is False
    # Default crag_max_rewrites=1 -> one rewrite -> two attempts total.
    assert out.correction.attempts == 2
