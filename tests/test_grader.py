"""Tests for CRAG retrieval grading.

We inject a fake LLM judge so these run without a live Ollama — the point is to
pin the two-tier logic (score pre-gate vs. LLM tie-break), not the model.
"""
from src.rag.config import settings
from src.rag.grader import grade_documents
from src.rag.vectorstore import Hit


def _hit(rerank_score=None, score=0.5):
    return Hit(text="body", score=score, source="x", rerank_score=rerank_score)


def test_no_hits_is_not_relevant():
    g = grade_documents("q", [], "", judge=lambda q, c: True)
    assert g.relevant is False
    assert g.used_llm is False


def test_high_score_pregate_skips_llm():
    high = settings.crag_grade_min_score + 0.1
    called = []
    judge = lambda q, c: called.append(1) or False  # would say "no" if called
    g = grade_documents("q", [_hit(rerank_score=high)], "ctx", judge=judge)
    assert g.relevant is True
    assert g.used_llm is False
    assert not called  # LLM judge never ran — the score alone decided


def test_low_score_defers_to_llm_yes():
    low = settings.crag_grade_min_score - 0.1
    g = grade_documents("q", [_hit(rerank_score=low)], "ctx", judge=lambda q, c: True)
    assert g.relevant is True
    assert g.used_llm is True


def test_low_score_defers_to_llm_no():
    low = settings.crag_grade_min_score - 0.1
    g = grade_documents("q", [_hit(rerank_score=low)], "ctx", judge=lambda q, c: False)
    assert g.relevant is False
    assert g.used_llm is True


def test_falls_back_to_cosine_when_no_rerank_score():
    # No rerank score -> pre-gate uses cosine; here cosine clears the gate.
    high = settings.crag_grade_min_score + 0.1
    g = grade_documents("q", [_hit(rerank_score=None, score=high)], "ctx",
                        judge=lambda q, c: False)
    assert g.relevant is True
    assert g.used_llm is False
