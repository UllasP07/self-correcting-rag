"""The RAG query pipeline.

    question -> retrieve -> rerank -> [CRAG: grade -> maybe rewrite+retry] -> generate

Milestones 1-3 built this as a straight line: retrieve, rerank, generate. If
retrieval was weak, the LLM answered from weak context anyway. Milestone 4 wraps
these same steps in a LangGraph state machine (see graph.py) that *grades* the
retrieval and, when it's too weak, rewrites the query and retries once before
answering.

This module now holds the reusable *steps* (retrieve, generate, build context)
that the graph orchestrates. `answer_question` is the public entry point; it
delegates to the CRAG graph when enabled, else runs the linear path.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .config import settings
from .llm import chat, embed_one, embedder_fingerprint
from .reranker import rerank
from .vectorstore import Hit, VectorStore

SYSTEM_PROMPT = (
    "You are an enterprise assistant. Answer the user's question using ONLY the "
    "provided context. If the context does not contain the answer, say you don't "
    "know rather than guessing. Cite the source filename in brackets like [file.pdf]."
)


@dataclass
class Correction:
    """Trace of the CRAG self-correction, so the CLI can SHOW what happened."""
    attempts: int = 1                       # how many retrieval rounds ran
    graded_relevant: bool = True            # final grade verdict
    grade_reason: str = ""                  # why the grader decided that
    rewritten_query: str | None = None      # the rewritten query, if we rewrote


@dataclass
class Answer:
    text: str
    hits: list[Hit]
    top_score: float
    correction: Correction = field(default_factory=Correction)


def _build_context(hits: list[Hit]) -> str:
    """Assemble the LLM context from the PARENT sections of the matched chunks.

    We matched on small children, but hand the model the fuller parent sections.
    Parents are deduped — several children can share one section, and we don't
    want to send the same text five times.
    """
    blocks: list[str] = []
    seen: set[str] = set()
    for h in hits:
        context = h.parent or h.text
        if context in seen:
            continue
        seen.add(context)
        header = f"[source: {h.source}"
        if h.title:
            header += f" | section: {h.title}"
        header += f" | score: {h.score:.3f}]"
        blocks.append(f"{header}\n{context}")
    return "\n\n---\n\n".join(blocks)


def _top_score(hits: list[Hit]) -> float:
    if not hits:
        return 0.0
    h = hits[0]
    return h.rerank_score if h.rerank_score is not None else h.score


def retrieve(query: str, store: VectorStore | None = None) -> list[Hit]:
    """Embed `query`, pull a wide candidate pool, then rerank down to TOP_K.

    This is one shared step used by both the linear path and every retrieval
    round of the CRAG graph. `store` is injectable so the graph can reuse one
    client across rounds.
    """
    store = store or VectorStore()
    qvec = embed_one(query)
    hits = store.search(qvec, top_k=settings.retrieve_k)
    if settings.rerank_enabled and hits:
        hits = rerank(query, hits, top_n=settings.top_k)
    else:
        hits = hits[: settings.top_k]
    return hits


def generate(question: str, hits: list[Hit]) -> str:
    """Build a grounded prompt from `hits` and generate the answer."""
    context = _build_context(hits) if hits else "(no documents retrieved)"
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    return chat(SYSTEM_PROMPT, user_prompt)


def _answer_linear(question: str) -> Answer:
    """The pre-M4 straight-line path (used when CRAG is disabled)."""
    hits = retrieve(question)
    text = generate(question, hits)
    return Answer(text=text, hits=hits, top_score=_top_score(hits))


def answer_question(question: str) -> Answer:
    # Guard: refuse if the query embedder differs from the one that built the
    # index. Cross-embedder similarity is meaningless — fail loudly rather than
    # answer from garbage. (Done once, up front, for both paths.)
    VectorStore().assert_embedder(embedder_fingerprint())

    if not settings.crag_enabled:
        return _answer_linear(question)

    # Lazy import: keeps langgraph off the import path when CRAG is disabled.
    from .graph import run_crag
    return run_crag(question)
