"""CRAG self-correcting loop as a LangGraph state machine (Milestone 4).

Milestones 1-3 were a straight line. The failure it hides: when retrieval is
weak, the LLM still confidently answers from weak context. CRAG (Corrective RAG)
fixes this by making retrieval a *loop with a judge*:

    retrieve -> grade -+-- relevant --------------------> generate
                       |
                       +-- weak & rewrites left --> rewrite_query --> (retrieve)
                       |
                       +-- weak & out of retries -------> generate  (answer honestly,
                                                                      likely "I don't know")

Why a graph and not more if/else? Because from here on the control flow branches
and loops (M5 adds a text-to-SQL branch, M6 a human-in-the-loop pause). A
StateGraph makes those explicit and inspectable instead of nested conditionals.

Every dependency (retrieve / grade / rewrite / generate) is INJECTABLE so the
whole loop can be tested without a live Ollama or Qdrant.
"""
from __future__ import annotations

from typing import Callable, Optional, TypedDict

from langgraph.graph import END, StateGraph

from .config import settings
from .grader import Grade, grade_documents
from .llm import chat
from .pipeline import Answer, Correction, _build_context, _top_score, generate, retrieve
from .vectorstore import Hit, VectorStore

REWRITE_SYSTEM = (
    "You rewrite a user's question into a better search query for a document "
    "retrieval system. Keep it concise, expand key terms and likely synonyms, "
    "and output ONLY the rewritten query with no preamble."
)


def _default_rewrite(question: str, _hits: list[Hit]) -> str:
    """Ask the LLM to rephrase the question for better retrieval."""
    return chat(REWRITE_SYSTEM, question).strip()


class CragState(TypedDict, total=False):
    question: str          # the original user question (never mutated)
    query: str             # current search query (rewritten on retry)
    hits: list[Hit]
    context: str
    grade: Grade
    rewrites: int          # how many times we've rewritten so far
    answer: str


def build_graph(
    retrieve_fn: Callable[[str], list[Hit]] = None,
    grade_fn: Callable[[str, list[Hit], str], Grade] = None,
    rewrite_fn: Callable[[str, list[Hit]], str] = None,
    generate_fn: Callable[[str, list[Hit]], str] = None,
):
    """Compile the CRAG graph. Args override the real steps (used by tests)."""
    # A single VectorStore reused across retrieval rounds (only for the real fn).
    store = None if retrieve_fn else VectorStore()
    retrieve_fn = retrieve_fn or (lambda q: retrieve(q, store=store))
    grade_fn = grade_fn or grade_documents
    rewrite_fn = rewrite_fn or _default_rewrite
    generate_fn = generate_fn or generate

    def retrieve_node(state: CragState) -> CragState:
        hits = retrieve_fn(state["query"])
        return {"hits": hits, "context": _build_context(hits) if hits else ""}

    def grade_node(state: CragState) -> CragState:
        grade = grade_fn(state["question"], state["hits"], state["context"])
        return {"grade": grade}

    def rewrite_node(state: CragState) -> CragState:
        new_query = rewrite_fn(state["question"], state["hits"])
        return {"query": new_query, "rewrites": state.get("rewrites", 0) + 1}

    def generate_node(state: CragState) -> CragState:
        return {"answer": generate_fn(state["question"], state["hits"])}

    def after_grade(state: CragState) -> str:
        """Branch: good context -> generate; weak but retries left -> rewrite."""
        if state["grade"].relevant:
            return "generate"
        if state.get("rewrites", 0) < settings.crag_max_rewrites:
            return "rewrite"
        return "generate"  # out of retries — answer honestly from what we have

    g = StateGraph(CragState)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grade_node)
    g.add_node("rewrite", rewrite_node)
    g.add_node("generate", generate_node)

    g.set_entry_point("retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges("grade", after_grade,
                            {"rewrite": "rewrite", "generate": "generate"})
    g.add_edge("rewrite", "retrieve")   # loop back with the rewritten query
    g.add_edge("generate", END)
    return g.compile()


def run_crag(question: str, **overrides) -> Answer:
    """Run the CRAG loop for `question` and package the result as an `Answer`.

    `overrides` are forwarded to `build_graph` so tests can inject fake steps.
    """
    graph = build_graph(**overrides)
    final: CragState = graph.invoke(
        {"question": question, "query": question, "rewrites": 0}
    )

    hits = final.get("hits", [])
    grade: Grade = final.get("grade", Grade(True, "", False))
    rewrites = final.get("rewrites", 0)
    return Answer(
        text=final.get("answer", ""),
        hits=hits,
        top_score=_top_score(hits),
        correction=Correction(
            attempts=rewrites + 1,
            graded_relevant=grade.relevant,
            grade_reason=grade.reason,
            # If we rewrote, `query` differs from the original question.
            rewritten_query=(final.get("query") if rewrites else None),
        ),
    )
