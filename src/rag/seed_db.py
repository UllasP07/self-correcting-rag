"""Seed a local SQLite database with sample structured data (Milestone 5).

Text-to-SQL needs an actual database to query. This builds a small, realistic
`employees` table themed to match the ACME handbook, so routing demos feel
natural: prose questions ("what's the PTO policy?") go to the document loop,
data questions ("how many people in Engineering?") go to the SQL node.

    python -m src.rag.seed_db          # (re)build data/acme.db

Idempotent: drops and recreates the table each run.
"""
from __future__ import annotations

import sqlite3

from rich.console import Console

from .database import Database

console = Console()

# (name, department, title, salary, hire_date, pto_days_used)
_EMPLOYEES = [
    ("Ava Chen",        "Engineering", "Staff Engineer",       182000, "2019-03-11", 6),
    ("Marcus Reed",     "Engineering", "Senior Engineer",      154000, "2020-07-01", 12),
    ("Priya Nair",      "Engineering", "Engineer",             128000, "2022-01-18", 3),
    ("Diego Santos",    "Engineering", "Engineer",             124000, "2023-05-02", 1),
    ("Lena Fischer",    "Engineering", "Engineering Manager",  198000, "2018-09-23", 9),
    ("Tom Whitfield",   "Sales",       "Account Executive",    112000, "2021-02-14", 14),
    ("Sofia Marino",    "Sales",       "Account Executive",    108000, "2022-11-07", 5),
    ("Grace Okafor",    "Sales",       "Sales Director",       176000, "2017-06-30", 11),
    ("Ben Kowalski",    "Marketing",   "Content Lead",         119000, "2020-10-12", 8),
    ("Hana Suzuki",     "Marketing",   "Marketing Manager",    141000, "2019-12-05", 7),
    ("Isabel Cruz",     "People",      "HR Business Partner",  126000, "2021-08-19", 10),
    ("Noah Bergstrom",  "People",      "Recruiter",             98000, "2023-01-09", 2),
    ("Wei Zhang",       "Finance",     "Financial Analyst",    115000, "2022-04-25", 4),
    ("Omar Haddad",     "Finance",     "Controller",           168000, "2018-03-16", 13),
    ("Ruth Alemu",      "Finance",     "Finance Manager",      152000, "2020-05-11", 6),
]

_DDL = """
CREATE TABLE employees (
    id            INTEGER PRIMARY KEY,
    name          TEXT    NOT NULL,
    department    TEXT    NOT NULL,
    title         TEXT    NOT NULL,
    salary        INTEGER NOT NULL,   -- annual USD
    hire_date     TEXT    NOT NULL,   -- ISO date YYYY-MM-DD
    pto_days_used INTEGER NOT NULL    -- vacation days taken this year
);
"""


def seed() -> None:
    db = Database()
    with sqlite3.connect(db.path) as conn:
        conn.execute("DROP TABLE IF EXISTS employees")
        conn.executescript(_DDL)
        conn.executemany(
            "INSERT INTO employees "
            "(name, department, title, salary, hire_date, pto_days_used) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            _EMPLOYEES,
        )
        conn.commit()
    console.print(f"[green]Seeded {len(_EMPLOYEES)} employees[/green] into {db.path}")


if __name__ == "__main__":
    seed()
