"""Tests for the CRAG graph's branching, with every step injected.

No Ollama, no Qdrant, no model downloads — we drive the state machine with fake
retrieve/grade/rewrite/generate functions and assert the control flow.
"""
import pytest

pytest.importorskip("langgraph")  # skip cleanly if the M4 dep isn't installed

from langgraph.checkpoint.memory import MemorySaver

from src.rag import approvals
from src.rag.grader import Grade
from src.rag.graph import resume_sql, run_crag, run_sql
from src.rag.risk import RiskAssessment
from src.rag.text_to_sql import SQLQuery
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


# --- text-to-SQL branch (M5) ---

def test_sql_branch_generates_executes_answers():
    out = run_sql(
        "how many engineers?",
        hitl=False,  # HITL exercised separately; keep this test hermetic
        schema="CREATE TABLE employees(department TEXT)",
        sql_fn=lambda q, s: SQLQuery(sql="SELECT COUNT(*) AS n FROM employees"),
        execute_fn=lambda sql: [{"n": 5}],
        answer_fn=lambda q, sql, rows: f"There are {rows[0]['n']}.",
    )
    assert out.route == "structured"
    assert out.text == "There are 5."
    assert out.sql == "SELECT COUNT(*) AS n FROM employees"
    assert out.row_count == 1


def test_sql_branch_handles_unsafe_generation_gracefully():
    # sql_fn raises (as SQLQuery would on unsafe output) -> error path, no execute.
    def boom_sql(q, s):
        raise ValueError("only SELECT queries are allowed")

    out = run_sql(
        "delete everyone",
        hitl=False,
        schema="schema",
        sql_fn=boom_sql,
        execute_fn=lambda sql: pytest.fail("must not execute after generation error"),
        answer_fn=lambda q, sql, rows: pytest.fail("must not answer from rows"),
    )
    assert out.route == "structured"
    assert out.sql is None
    assert "couldn't answer" in out.text.lower()


# --- HITL freeze / approve / deny (M6) ---

def _sql_overrides(risky):
    """Injected SQL steps + a canned risk verdict, so no LLM/DB/real files."""
    reasons = ["reads sensitive column 'salary'"] if risky else []
    return dict(
        schema="CREATE TABLE employees(salary INTEGER)",
        sql_fn=lambda q, s: SQLQuery(sql="SELECT salary FROM employees WHERE id=1"),
        execute_fn=lambda sql: [{"salary": 100}],
        answer_fn=lambda q, sql, rows: f"salary is {rows[0]['salary']}",
        risk_fn=lambda sql: RiskAssessment(risky=risky, reasons=reasons),
        hitl=True,
    )


@pytest.fixture(autouse=True)
def _no_approvals_file(monkeypatch):
    # Keep the persistent approvals registry out of tests.
    monkeypatch.setattr(approvals, "record_pending", lambda *a, **k: None)
    monkeypatch.setattr(approvals, "set_status", lambda *a, **k: None)


def test_safe_query_auto_executes_under_hitl():
    out = run_sql("who?", request_id="s1", checkpointer=MemorySaver(),
                  **_sql_overrides(risky=False))
    assert out.status == "executed"
    assert out.text == "salary is 100"


def test_risky_query_freezes():
    out = run_sql("salary?", request_id="f1", checkpointer=MemorySaver(),
                  **_sql_overrides(risky=True))
    assert out.status == "frozen"
    assert out.sql == "SELECT salary FROM employees WHERE id=1"
    assert "salary" in out.risk_reason
    assert out.text == ""  # not executed — no answer yet


def test_frozen_then_approved_executes():
    saver = MemorySaver()
    ov = _sql_overrides(risky=True)
    frozen = run_sql("salary?", request_id="a1", checkpointer=saver, **ov)
    assert frozen.status == "frozen"
    resumed = resume_sql("a1", approved=True, checkpointer=saver, **ov)
    assert resumed.status == "executed"
    assert resumed.text == "salary is 100"


def test_frozen_then_denied_does_not_execute():
    saver = MemorySaver()
    ov = dict(_sql_overrides(risky=True),
              execute_fn=lambda sql: pytest.fail("denied query must not execute"))
    frozen = run_sql("salary?", request_id="d1", checkpointer=saver, **ov)
    assert frozen.status == "frozen"
    resumed = resume_sql("d1", approved=False, checkpointer=saver, **ov)
    assert resumed.status == "denied"
    assert "denied" in resumed.text.lower()
