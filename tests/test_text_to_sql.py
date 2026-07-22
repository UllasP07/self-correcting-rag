"""Tests for text-to-SQL schema enforcement.

The point is the *validation boundary*: the LLM's output must become a safe,
single read-only SELECT or fail to construct. We inject canned LLM output so no
Ollama is needed.
"""
import pytest
from pydantic import ValidationError

from src.rag.text_to_sql import SQLQuery, generate_sql


def test_accepts_plain_select():
    q = SQLQuery(sql="SELECT COUNT(*) FROM employees")
    assert q.sql == "SELECT COUNT(*) FROM employees"


def test_strips_markdown_fences_and_semicolon():
    q = SQLQuery(sql="```sql\nSELECT name FROM employees;\n```")
    assert q.sql == "SELECT name FROM employees"


def test_accepts_cte_with_statement():
    q = SQLQuery(sql="WITH x AS (SELECT 1) SELECT * FROM x")
    assert q.sql.lower().startswith("with")


@pytest.mark.parametrize("bad", [
    "DELETE FROM employees",
    "UPDATE employees SET salary = 0",
    "INSERT INTO employees VALUES (1)",
    "DROP TABLE employees",
    "SELECT 1; DROP TABLE employees",          # stacked statements
    "SELECT * FROM employees; DELETE FROM x",
    "",
])
def test_rejects_unsafe_sql(bad):
    with pytest.raises(ValidationError):
        SQLQuery(sql=bad)


def test_generate_sql_uses_injected_llm():
    llm = lambda system, user: "SELECT department, COUNT(*) FROM employees GROUP BY department"
    q = generate_sql("headcount per department", "CREATE TABLE employees(...)", llm=llm)
    assert q.sql.startswith("SELECT department")


def test_generate_sql_rejects_unsafe_model_output():
    llm = lambda system, user: "DROP TABLE employees"
    with pytest.raises(ValidationError):
        generate_sql("delete everything", "schema", llm=llm)
