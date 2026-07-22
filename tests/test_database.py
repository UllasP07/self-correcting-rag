"""Tests for the SQLite wrapper, against a real temp database (no services)."""
import sqlite3

import pytest

from src.rag.database import Database, NotReadOnlyError


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "test.db"
    with sqlite3.connect(path) as conn:
        conn.executescript(
            "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, department TEXT);"
            "INSERT INTO employees (name, department) VALUES ('Ava', 'Engineering');"
            "INSERT INTO employees (name, department) VALUES ('Ben', 'Sales');"
        )
    return Database(str(path))


def test_schema_ddl_exposes_table(db):
    ddl = db.schema_ddl()
    assert "CREATE TABLE employees" in ddl
    assert "department" in ddl


def test_run_select_returns_dicts(db):
    rows = db.run_select("SELECT name, department FROM employees ORDER BY name")
    assert rows == [
        {"name": "Ava", "department": "Engineering"},
        {"name": "Ben", "department": "Sales"},
    ]


def test_run_select_respects_max_rows(db):
    rows = db.run_select("SELECT * FROM employees", max_rows=1)
    assert len(rows) == 1


@pytest.mark.parametrize("bad", [
    "DELETE FROM employees",
    "SELECT 1; DROP TABLE employees",
])
def test_run_select_rejects_non_select(db, bad):
    with pytest.raises(NotReadOnlyError):
        db.run_select(bad)


def test_read_only_uri_blocks_writes(db):
    # run_select opens the DB with mode=ro; a write through that URI must fail
    # at the SQLite layer (defense in depth behind the SELECT-only check).
    con = sqlite3.connect(f"file:{db.path}?mode=ro", uri=True)
    with pytest.raises(sqlite3.OperationalError):
        con.execute("DELETE FROM employees")
    con.close()
