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

import sqlite3
import uuid
from pathlib import Path
from typing import Callable, Optional, TypedDict

from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from . import approvals
from .config import settings
from .database import Database
from .grader import Grade, grade_documents
from .llm import chat
from .pipeline import Answer, Correction, _build_context, _top_score, generate, retrieve
from .risk import RiskAssessment, assess_risk
from .text_to_sql import SQLQuery, generate_sql
from .vectorstore import Hit, VectorStore

_ROOT = Path(__file__).resolve().parents[2]

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
        route="documents",
        correction=Correction(
            attempts=rewrites + 1,
            graded_relevant=grade.relevant,
            grade_reason=grade.reason,
            # If we rewrote, `query` differs from the original question.
            rewritten_query=(final.get("query") if rewrites else None),
        ),
    )


# --------------------------------------------------------------------------
# Text-to-SQL branch (Milestone 5)
# --------------------------------------------------------------------------

SQL_ANSWER_SYSTEM = (
    "You answer the user's question using ONLY the SQL query result rows below. "
    "Be concise and state the numbers directly. If the rows are empty, say the "
    "data doesn't contain an answer."
)


def _default_sql_answer(question: str, sql: str, rows: list[dict]) -> str:
    """Turn result rows into a natural-language answer."""
    return chat(
        SQL_ANSWER_SYSTEM,
        f"Question: {question}\n\nSQL:\n{sql}\n\nRows:\n{rows}",
    )


class SqlState(TypedDict, total=False):
    question: str
    schema: str
    sql: str            # the validated SQL that ran (empty if generation failed)
    rows: list[dict]
    error: str          # set if generation/validation/execution failed
    answer: str
    risk_reasons: list[str]   # why the query was flagged (M6); [] if safe
    approved: bool            # admin decision after a freeze (M6)


def _sqlite_checkpointer():
    """A durable checkpointer so frozen queries survive a process restart."""
    from langgraph.checkpoint.sqlite import SqliteSaver

    raw = Path(settings.checkpoint_path)
    path = str(raw if raw.is_absolute() else _ROOT / raw)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, check_same_thread=False)
    return SqliteSaver(conn)


def build_sql_graph(
    schema: str = None,
    sql_fn: Callable[[str, str], SQLQuery] = None,
    execute_fn: Callable[[str], list[dict]] = None,
    answer_fn: Callable[[str, str, list[dict]], str] = None,
    risk_fn: Callable[[str], RiskAssessment] = None,
    hitl: bool | None = None,
    checkpointer=None,
):
    """Compile the text-to-SQL graph. Args override real steps (used by tests).

    When `hitl` is on, a risky query pauses at the `approval` node via a durable
    `interrupt()`; resume it with `Command(resume={"approved": bool})`.
    """
    db = None if (schema is not None and execute_fn) else Database()
    schema = schema if schema is not None else db.schema_ddl()
    sql_fn = sql_fn or generate_sql
    execute_fn = execute_fn or (lambda s: db.run_select(s))
    answer_fn = answer_fn or _default_sql_answer
    risk_fn = risk_fn or assess_risk
    hitl = settings.hitl_enabled if hitl is None else hitl
    # interrupt() requires a checkpointer to persist the paused state.
    if hitl and checkpointer is None:
        checkpointer = _sqlite_checkpointer()

    def generate_sql_node(state: SqlState) -> SqlState:
        # Validation lives in the SQLQuery model; a bad/unsafe query raises here.
        try:
            query = sql_fn(state["question"], schema)
            return {"sql": query.sql}
        except Exception as e:  # noqa: BLE001 — surface as a graceful error node
            return {"error": f"could not generate a safe query: {e}"}

    def assess_node(state: SqlState) -> SqlState:
        # Runs to completion BEFORE any interrupt, so the reasons are checkpointed
        # and an admin can see why the query was flagged.
        return {"risk_reasons": risk_fn(state["sql"]).reasons}

    def approval_node(state: SqlState) -> SqlState:
        if not state.get("risk_reasons"):
            return {"approved": True}          # safe → auto-approve
        # Risky → freeze here. interrupt() halts the graph and persists state;
        # the value passed to Command(resume=...) is what it returns on resume.
        decision = interrupt({"sql": state["sql"], "reasons": state["risk_reasons"]})
        approved = decision.get("approved") if isinstance(decision, dict) else bool(decision)
        if not approved:
            return {"approved": False, "error": "query was denied by an administrator"}
        return {"approved": True}

    def execute_node(state: SqlState) -> SqlState:
        try:
            return {"rows": execute_fn(state["sql"])}
        except Exception as e:  # noqa: BLE001
            return {"error": f"query failed to execute: {e}"}

    def answer_node(state: SqlState) -> SqlState:
        if state.get("error"):
            return {"answer": f"I couldn't answer that from the database ({state['error']})."}
        return {"answer": answer_fn(state["question"], state["sql"], state.get("rows", []))}

    g = StateGraph(SqlState)
    g.add_node("generate_sql", generate_sql_node)
    g.add_node("execute", execute_node)
    g.add_node("answer", answer_node)
    g.set_entry_point("generate_sql")
    g.add_edge("execute", "answer")
    g.add_edge("answer", END)

    if hitl:
        g.add_node("assess", assess_node)
        g.add_node("approval", approval_node)
        g.add_conditional_edges(
            "generate_sql", lambda s: "answer" if s.get("error") else "assess",
            {"assess": "assess", "answer": "answer"})
        g.add_edge("assess", "approval")
        g.add_conditional_edges(
            "approval", lambda s: "execute" if s.get("approved") else "answer",
            {"execute": "execute", "answer": "answer"})
    else:
        g.add_conditional_edges(
            "generate_sql", lambda s: "answer" if s.get("error") else "execute",
            {"execute": "execute", "answer": "answer"})

    return g.compile(checkpointer=checkpointer)


def _sql_answer(final: SqlState, request_id: str, status: str) -> Answer:
    return Answer(
        text=final.get("answer", ""),
        hits=[],
        top_score=0.0,
        route="structured",
        sql=final.get("sql") or None,
        row_count=len(final.get("rows", [])),
        status=status,
        request_id=request_id,
        risk_reason="; ".join(final.get("risk_reasons", [])),
    )


def run_sql(question: str, request_id: str | None = None, **overrides) -> Answer:
    """Run the text-to-SQL graph. Returns a *frozen* Answer if it hit the gate."""
    graph = build_sql_graph(**overrides)
    request_id = request_id or uuid.uuid4().hex[:8]

    # No checkpointer (HITL off) → straight-through run, never freezes.
    if graph.checkpointer is None:
        final: SqlState = graph.invoke({"question": question})
        return _sql_answer(final, request_id, status="executed")

    config = {"configurable": {"thread_id": request_id}}
    graph.invoke({"question": question}, config=config)
    snap = graph.get_state(config)
    final = snap.values
    if snap.next:  # pending node => paused at the approval gate
        approvals.record_pending(
            request_id, question, final.get("sql", ""),
            "; ".join(final.get("risk_reasons", [])))
        return _sql_answer(final, request_id, status="frozen")
    return _sql_answer(final, request_id, status="executed")


def resume_sql(request_id: str, approved: bool, **overrides) -> Answer:
    """Resume a frozen SQL request with the admin's decision (approve/deny)."""
    graph = build_sql_graph(**overrides)
    config = {"configurable": {"thread_id": request_id}}
    graph.invoke(Command(resume={"approved": approved}), config=config)
    approvals.set_status(request_id, "approved" if approved else "denied")
    final: SqlState = graph.get_state(config).values
    return _sql_answer(final, request_id, status="executed" if approved else "denied")
