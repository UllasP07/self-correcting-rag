"""Text-to-SQL with Pydantic schema enforcement (Milestone 5).

THE MILESTONE'S POINT: an LLM writing SQL is useful but untrusted. It can emit
prose, markdown fences, or (worse) a destructive statement. So we don't run its
raw text — the model emits into a *typed contract* (`SQLQuery`) that we validate
before anything touches the database:

  1. strip markdown/prose noise the model often wraps SQL in,
  2. enforce a SINGLE read-only SELECT (no INSERT/UPDATE/DELETE/DROP, no stacked
     statements) — a validator on the Pydantic model,
  3. only then hand the `.sql` string to the database layer (which re-checks).

This typed, validated boundary is exactly what Milestone 6 hangs the
human-approval gate on: a `SQLQuery` is a reviewable object, not a blob of text.
"""
from __future__ import annotations

import re
from typing import Callable

from pydantic import BaseModel, field_validator

from .llm import chat

_WRITE_KEYWORDS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|truncate|attach|pragma)\b",
    re.IGNORECASE,
)


class SQLQuery(BaseModel):
    """A validated, read-only SQL query. Construction fails if it isn't safe."""
    sql: str

    @field_validator("sql")
    @classmethod
    def _must_be_read_only_select(cls, v: str) -> str:
        sql = _strip_fences(v).strip().rstrip(";").strip()
        if not sql:
            raise ValueError("empty SQL")
        if ";" in sql:
            raise ValueError("only a single statement is allowed")
        if not sql.lower().startswith(("select", "with")):
            raise ValueError("only SELECT queries are allowed")
        if _WRITE_KEYWORDS.search(sql):
            raise ValueError("query contains a write/DDL keyword")
        return sql


def _strip_fences(text: str) -> str:
    """Remove ```sql ... ``` fences and stray backticks the LLM often adds."""
    text = text.strip()
    fence = re.match(r"^```(?:sql)?\s*(.*?)\s*```$", text, re.DOTALL | re.IGNORECASE)
    if fence:
        text = fence.group(1)
    return text.strip("` \n")


SQL_SYSTEM = (
    "You translate a question into ONE read-only SQLite SELECT query. "
    "Use only the tables and columns in the provided schema. Return ONLY the SQL "
    "query — no explanation, no markdown fences, no trailing semicolon."
)


def generate_sql(
    question: str,
    schema: str,
    llm: Callable[[str, str], str] | None = None,
) -> SQLQuery:
    """Ask the LLM for SQL and validate it into a `SQLQuery` (raises on unsafe).

    `llm` is injectable so tests can supply canned model output.
    """
    llm = llm or chat
    raw = llm(SQL_SYSTEM, f"Schema:\n{schema}\n\nQuestion: {question}")
    return SQLQuery(sql=raw)
