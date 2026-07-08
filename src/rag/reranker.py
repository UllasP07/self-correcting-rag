"""Cross-encoder reranking (Milestone 3).

WHY: Qdrant's cosine search is a *bi-encoder* — it embeds the query and each
chunk separately, then compares. Fast, but it never actually reads the query and
chunk *together*, so it mis-ranks subtle cases (a generic intro can out-score the
precise section).

A *cross-encoder* (BGE-Reranker) takes (query, chunk) as ONE input and outputs a
relevance score. Much more accurate — but too slow to run over the whole DB. So
the standard pattern is two-stage:

    retrieve a wide pool with the fast bi-encoder  →  rerank the pool with the
    slow-but-accurate cross-encoder  →  keep the top few.

The model is lazy-loaded on first use, and the scorer is injectable so tests can
run without downloading PyTorch/the model.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

from .config import settings
from .vectorstore import Hit

_model = None


def _load():
    global _model
    if _model is None:
        from sentence_transformers import CrossEncoder
        _model = CrossEncoder(settings.rerank_model)
    return _model


def _default_scorer(query: str, texts: Sequence[str]) -> list[float]:
    """Score each (query, text) pair with the BGE cross-encoder → [0, 1]."""
    model = _load()
    raw = model.predict([(query, t) for t in texts])
    # BGE outputs a raw logit; squash to [0,1] with a sigmoid for readability.
    return [1.0 / (1.0 + math.exp(-float(r))) for r in raw]


def rerank(
    query: str,
    hits: list[Hit],
    top_n: int | None = None,
    scorer: Callable[[str, Sequence[str]], list[float]] | None = None,
) -> list[Hit]:
    """Re-score `hits` against `query` and return the best `top_n`, reordered.

    We score against the CHILD text (`hit.text`) — the precise snippet that
    matched — not the fuller parent.
    """
    top_n = top_n or settings.top_k
    if not hits:
        return hits
    scorer = scorer or _default_scorer
    scores = scorer(query, [h.text for h in hits])
    for h, s in zip(hits, scores):
        h.rerank_score = float(s)
    hits.sort(key=lambda h: (h.rerank_score if h.rerank_score is not None else -1.0),
              reverse=True)
    return hits[:top_n]
