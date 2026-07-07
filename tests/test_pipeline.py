"""Tests for pure pipeline helpers (no services needed)."""
from src.rag.pipeline import _build_context
from src.rag.vectorstore import Hit


def test_build_context_formats_and_separates():
    hits = [
        Hit(text="alpha body", score=0.9, source="a.md"),
        Hit(text="beta body", score=0.512, source="b.md"),
    ]
    ctx = _build_context(hits)
    # both sources + texts present
    assert "a.md" in ctx and "b.md" in ctx
    assert "alpha body" in ctx and "beta body" in ctx
    # scores rendered to 3 decimals
    assert "0.900" in ctx and "0.512" in ctx
    # blocks separated
    assert "---" in ctx


def test_build_context_empty():
    assert _build_context([]) == ""
