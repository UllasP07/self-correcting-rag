"""Retrieval grading (Milestone 4, CRAG).

The whole point of CRAG (Corrective RAG) is that the system should *notice* when
its own retrieval is weak instead of blindly generating from junk. This module is
that noticing step: given the question and the retrieved hits, decide whether the
context is good enough to answer from.

TWO-TIER GRADING (cheap → expensive):
  1. Fast pre-gate on the top score. If the best (rerank) score comfortably
     clears a threshold, the retrieval is obviously good — return "relevant"
     without spending an LLM call.
  2. Otherwise, ask the LLM to actually read (question, context) and judge
     relevance. Cross-encoder scores are only a proxy; the LLM is the tie-breaker
     for the murky middle.

The LLM judge is INJECTABLE (like the reranker's scorer) so tests can exercise
the branching logic without a live Ollama.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .config import settings
from .llm import chat
from .vectorstore import Hit

GRADE_SYSTEM = (
    "You are a strict retrieval grader. Decide whether the provided context "
    "contains enough information to answer the question. Reply with a single "
    "word: 'yes' if it does, 'no' if it does not. Do not explain."
)


@dataclass
class Grade:
    """Verdict on a retrieval attempt."""
    relevant: bool
    reason: str            # short, human-readable trace of *why* we decided
    used_llm: bool         # True if the LLM judge ran (False if score pre-gate decided)


def _top_score(hits: list[Hit]) -> float:
    """Best available score: prefer the cross-encoder rerank score, else cosine."""
    if not hits:
        return 0.0
    h = hits[0]
    return h.rerank_score if h.rerank_score is not None else h.score


def _llm_judge(question: str, context: str) -> bool:
    """Ask the chat model whether `context` can answer `question`."""
    reply = chat(GRADE_SYSTEM, f"Question: {question}\n\nContext:\n{context}")
    return reply.strip().lower().startswith("y")


def grade_documents(
    question: str,
    hits: list[Hit],
    context: str,
    judge: Callable[[str, str], bool] | None = None,
) -> Grade:
    """Grade whether `hits`/`context` are sufficient to answer `question`.

    `judge` overrides the LLM judge (for tests). `context` is the already-built
    prompt context so the judge sees exactly what the generator will.
    """
    if not hits:
        return Grade(relevant=False, reason="no documents retrieved", used_llm=False)

    top = _top_score(hits)
    if top >= settings.crag_grade_min_score:
        return Grade(
            relevant=True,
            reason=f"top score {top:.3f} >= {settings.crag_grade_min_score} (pre-gate)",
            used_llm=False,
        )

    judge = judge or _llm_judge
    ok = judge(question, context)
    return Grade(
        relevant=ok,
        reason=f"top score {top:.3f} below gate; LLM judge said {'yes' if ok else 'no'}",
        used_llm=True,
    )
