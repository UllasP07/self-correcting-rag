"""Tests for the reranker's reorder/truncate logic.

We inject a fake scorer, so these run without PyTorch or downloading the BGE
model — the point is to pin the *ordering* behavior, not the model's accuracy.
"""
from src.rag.reranker import rerank
from src.rag.vectorstore import Hit


def _hits():
    # cosine order is a > b > c
    return [
        Hit(text="a", score=0.9, source="x"),
        Hit(text="b", score=0.8, source="x"),
        Hit(text="c", score=0.7, source="x"),
    ]


def test_rerank_reorders_by_cross_encoder_score():
    # cross-encoder flips the order: c most relevant, a least
    fake = lambda q, texts: [{"a": 0.1, "b": 0.2, "c": 0.9}[t] for t in texts]
    out = rerank("q", _hits(), top_n=3, scorer=fake)
    assert [h.text for h in out] == ["c", "b", "a"]
    assert out[0].rerank_score == 0.9
    # original cosine score is preserved alongside the new rerank score
    assert out[0].score == 0.7


def test_rerank_truncates_to_top_n():
    fake = lambda q, texts: [0.5, 0.9, 0.1]  # a=0.5, b=0.9, c=0.1
    out = rerank("q", _hits(), top_n=2, scorer=fake)
    assert [h.text for h in out] == ["b", "a"]
    assert len(out) == 2


def test_rerank_empty_is_noop():
    assert rerank("q", [], scorer=lambda q, t: []) == []
