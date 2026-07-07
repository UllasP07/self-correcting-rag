"""Tests for the document loaders (.md, .txt, .xlsx) + dispatch.

Loaders are where messy real-world data breaks, so pin the behavior. Fixtures
are written to pytest's tmp_path — no repo files touched.
"""
import pytest
from openpyxl import Workbook

from src.rag.loaders import load_document, load_xlsx


def test_load_md(tmp_path):
    p = tmp_path / "doc.md"
    p.write_text("# Title\n\nsome body text")
    out = load_document(p)
    assert "some body text" in out


def test_load_txt(tmp_path):
    p = tmp_path / "doc.txt"
    p.write_text("plain content here")
    assert load_document(p) == "plain content here"


def test_load_xlsx_flattens_rows(tmp_path):
    wb = Workbook()
    ws = wb.active
    ws.title = "People"
    ws.append(["name", "age"])
    ws.append(["alice", 30])
    ws.append(["bob", 25])
    p = tmp_path / "data.xlsx"
    wb.save(p)

    out = load_xlsx(p)
    assert "People" in out              # sheet name is included
    assert "name=alice" in out          # header=value flattening
    assert "age=30" in out
    assert "name=bob" in out


def test_unsupported_extension_raises(tmp_path):
    p = tmp_path / "archive.zip"
    p.write_text("x")
    with pytest.raises(ValueError):
        load_document(p)
