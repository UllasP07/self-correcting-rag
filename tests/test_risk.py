"""Tests for the SQL risk policy (Milestone 6)."""
from src.rag.risk import assess_risk

COLS = ("salary",)


def test_flags_sensitive_column():
    r = assess_risk("SELECT name, salary FROM employees WHERE id = 1",
                    sensitive_columns=COLS)
    assert r.risky is True
    assert any("salary" in reason for reason in r.reasons)


def test_flags_unbounded_scan():
    r = assess_risk("SELECT name FROM employees", sensitive_columns=COLS)
    assert r.risky is True
    assert any("unbounded" in reason for reason in r.reasons)


def test_aggregate_is_not_a_broad_scan():
    # No WHERE/LIMIT, but it's an aggregate rollup, not a row dump.
    r = assess_risk("SELECT department, COUNT(*) FROM employees GROUP BY department",
                    sensitive_columns=COLS)
    assert r.risky is False


def test_filtered_non_sensitive_query_is_safe():
    r = assess_risk("SELECT name FROM employees WHERE department = 'Sales'",
                    sensitive_columns=COLS)
    assert r.risky is False


def test_limited_query_is_safe():
    r = assess_risk("SELECT name FROM employees LIMIT 10", sensitive_columns=COLS)
    assert r.risky is False


def test_broad_scan_flag_can_be_disabled():
    r = assess_risk("SELECT name FROM employees", sensitive_columns=COLS,
                    flag_broad_scans=False)
    assert r.risky is False


def test_sensitive_match_is_word_bounded():
    # 'salary_band' should not trip the 'salary' rule via substring.
    r = assess_risk("SELECT salary_band FROM grades WHERE id = 1",
                    sensitive_columns=COLS, flag_broad_scans=False)
    assert r.risky is False
