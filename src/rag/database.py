"""SQLite access for the text-to-SQL node (Milestone 5).

Deliberately tiny: two things the SQL node needs —
  - `schema_ddl()`  : the CREATE TABLE text we show the LLM so it can write SQL
                      against real columns (grounding the model in the schema).
  - `run_select()`  : execute a read-only query and return rows as dicts.

`run_select` is the LAST line of defense: even if a bad statement slipped past
the Pydantic validator, we open the connection read-only and reject anything
that isn't a single SELECT. Belt and suspenders — the validator is the belt.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from .config import settings

# Project root = two levels up from this file (src/rag/database.py).
_ROOT = Path(__file__).resolve().parents[2]


class NotReadOnlyError(Exception):
    """Raised when a query is anything other than a single read-only SELECT."""


class Database:
    def __init__(self, path: str | None = None) -> None:
        raw = Path(path or settings.sqlite_path)
        # Resolve relative paths against the project root, not the CWD.
        self.path = str(raw if raw.is_absolute() else _ROOT / raw)

    def exists(self) -> bool:
        return Path(self.path).exists()

    def schema_ddl(self) -> str:
        """Return the CREATE statements for all user tables (what we show the LLM)."""
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                "SELECT sql FROM sqlite_master "
                "WHERE type='table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL"
            ).fetchall()
        return "\n\n".join(r[0] for r in rows)

    @staticmethod
    def _assert_read_only(sql: str) -> None:
        stripped = sql.strip().rstrip(";").strip()
        if ";" in stripped:
            raise NotReadOnlyError("multiple statements are not allowed")
        if not stripped.lower().startswith("select"):
            raise NotReadOnlyError("only SELECT queries are allowed")

    def run_select(self, sql: str, max_rows: int | None = None) -> list[dict]:
        """Execute a single SELECT and return up to `max_rows` rows as dicts."""
        self._assert_read_only(sql)
        limit = max_rows or settings.sql_max_rows
        # file: URI with mode=ro makes SQLite itself refuse any write.
        uri = f"file:{self.path}?mode=ro"
        with sqlite3.connect(uri, uri=True) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(sql)
            rows = cur.fetchmany(limit)
            return [dict(r) for r in rows]
