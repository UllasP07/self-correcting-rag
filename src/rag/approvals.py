"""Persistent registry of frozen (awaiting-approval) SQL requests (Milestone 6).

LangGraph's checkpointer already persists the *graph state* needed to resume a
frozen run. But an admin also needs a human-facing list: "what's waiting, what
SQL, why was it flagged?" That's what this tiny table provides.

It lives in the same SQLite file as the checkpoints (a separate table), so the
whole HITL state — resumable graph + the approval queue — is one durable file
that survives a process restart.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import settings

_ROOT = Path(__file__).resolve().parents[2]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS approvals (
    request_id  TEXT PRIMARY KEY,
    question    TEXT NOT NULL,
    sql         TEXT NOT NULL,
    reasons     TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',   -- pending | approved | denied
    created_at  TEXT NOT NULL
);
"""


@dataclass
class ApprovalRecord:
    request_id: str
    question: str
    sql: str
    reasons: str
    status: str
    created_at: str


def _path() -> str:
    raw = Path(settings.checkpoint_path)
    return str(raw if raw.is_absolute() else _ROOT / raw)


def _connect() -> sqlite3.Connection:
    path = _path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(_SCHEMA)
    conn.row_factory = sqlite3.Row
    return conn


def record_pending(request_id: str, question: str, sql: str, reasons: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO approvals "
            "(request_id, question, sql, reasons, status, created_at) "
            "VALUES (?, ?, ?, ?, 'pending', ?)",
            (request_id, question, sql, reasons,
             datetime.now(timezone.utc).isoformat(timespec="seconds")),
        )
        conn.commit()


def set_status(request_id: str, status: str) -> None:
    with _connect() as conn:
        conn.execute("UPDATE approvals SET status = ? WHERE request_id = ?",
                     (status, request_id))
        conn.commit()


def get(request_id: str) -> ApprovalRecord | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM approvals WHERE request_id = ?",
                           (request_id,)).fetchone()
    return ApprovalRecord(**dict(row)) if row else None


def list_pending() -> list[ApprovalRecord]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM approvals WHERE status = 'pending' ORDER BY created_at"
        ).fetchall()
    return [ApprovalRecord(**dict(r)) for r in rows]
