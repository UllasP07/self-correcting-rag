"""The RAG query pipeline (Milestone 1) — a plain function, no graph yet.

    question -> embed -> Qdrant.search -> build context -> LLM -> answer

This is the classic "retrieve-then-generate" loop. Notice everything is linear:
one path, no branching. That is exactly why it is fragile — if search returns
junk, the LLM answers from junk. Milestone 4 turns this into a LangGraph that can
*grade* the retrieval and branch to a fallback. Feel the limitation here first.
"""
from __future__ import annotations

from dataclasses import dataclass

from .llm import chat, embed_one, embedder_fingerprint
from .vectorstore import Hit, VectorStore

SYSTEM_PROMPT = (
    "You are an enterprise assistant. Answer the user's question using ONLY the "
    "provided context. If the context does not contain the answer, say you don't "
    "know rather than guessing. Cite the source filename in brackets like [file.pdf]."
)


@dataclass
class Answer:
    text: str
    hits: list[Hit]
    top_score: float


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


def answer_question(question: str) -> Answer:
    store = VectorStore()

    # 0. Guard: refuse if the query embedder differs from the one that built the
    #    index (e.g. ingested with Ollama, now querying with Azure). Cross-embedder
    #    similarity is meaningless — fail loudly rather than answer from garbage.
    store.assert_embedder(embedder_fingerprint())

    # 1. Embed the question into the same vector space as the chunks.
    qvec = embed_one(question)

    # 2. Retrieve the nearest chunks.
    hits = store.search(qvec)
    top_score = hits[0].score if hits else 0.0

    # 3. Stuff them into a grounded prompt and generate.
    context = _build_context(hits) if hits else "(no documents retrieved)"
    user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
    text = chat(SYSTEM_PROMPT, user_prompt)

    return Answer(text=text, hits=hits, top_score=top_score)
