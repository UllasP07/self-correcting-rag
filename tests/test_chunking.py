"""Unit tests for the recursive character chunker (Milestone 1.5).

Pure logic, no running services needed — this is the highest-value thing to test
early because the recursive splitter is fiddly and everything downstream (embed,
retrieve, answer) is only as good as the chunks it produces.

Run:  python -m pytest -q
"""
from src.rag.chunking import (
    _SEPARATORS,
    _add_overlap,
    _split,
    chunk_text,
    parent_child_chunks,
    split_markdown_sections,
)
from src.rag.config import settings


# --- _split -------------------------------------------------------------

def test_split_returns_whole_text_when_it_fits():
    assert _split("short text", _SEPARATORS, 100) == ["short text"]


def test_split_prefers_paragraph_boundaries():
    text = "para one." + "\n\n" + "para two." + "\n\n" + "para three."
    # chunk_size small enough that each paragraph is its own chunk
    chunks = _split(text, _SEPARATORS, 12)
    assert chunks == ["para one.", "para two.", "para three."]


def test_split_hard_cuts_when_no_separator_left():
    # No separators available -> fall back to fixed-width character slices.
    assert _split("x" * 25, [""], 10) == ["x" * 10, "x" * 10, "x" * 5]


def test_split_recurses_into_oversized_part():
    # One giant word longer than chunk_size must still be broken up.
    chunks = _split("a" * 30, _SEPARATORS, 10)
    assert all(len(c) <= 10 for c in chunks)
    assert "".join(chunks) == "a" * 30


# --- _add_overlap -------------------------------------------------------

def test_add_overlap_prepends_previous_tail():
    assert _add_overlap(["abcdef", "ghijkl"], 3) == ["abcdef", "def ghijkl"]


def test_add_overlap_noop_for_single_chunk():
    assert _add_overlap(["only"], 3) == ["only"]


def test_add_overlap_zero_is_noop():
    assert _add_overlap(["abc", "def"], 0) == ["abc", "def"]


# --- chunk_text (public API, uses configured size/overlap) --------------

def test_chunk_text_empty_returns_empty():
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []


def test_chunk_text_short_stays_single():
    assert chunk_text("a single short paragraph") == ["a single short paragraph"]


def test_chunk_text_long_splits_and_bounds_size():
    text = ("Sentence number %d. " % 0) + " ".join(
        f"Sentence number {i}." for i in range(1, 400)
    )
    chunks = chunk_text(text)
    assert len(chunks) > 1
    # Each chunk is bounded by chunk_size, plus at most one overlap prefix + space.
    ceiling = settings.chunk_size + settings.chunk_overlap + 1
    assert all(len(c) <= ceiling for c in chunks)


# --- M2: structure-aware parent-child ----------------------------------

MD = "# Doc\n\nintro line\n\n## Alpha\n\nalpha body text\n\n## Beta\n\nbeta body text"


def test_split_markdown_sections_by_heading():
    sections = split_markdown_sections(MD)
    titles = [t for t, _ in sections]
    assert titles == ["Doc", "Alpha", "Beta"]
    # each section keeps its heading line for self-describing context
    assert sections[1][1].startswith("## Alpha")


def test_parent_child_makes_one_unit_per_small_section():
    units = parent_child_chunks(MD)
    titles = [u.title for u in units]
    assert "Alpha" in titles and "Beta" in titles
    # a small section's child IS its parent (no further split needed)
    beta = next(u for u in units if u.title == "Beta")
    assert "beta body text" in beta.child
    assert "beta body text" in beta.parent


def test_parent_child_falls_back_when_no_headings():
    plain = "just some text with no markdown headings at all"
    units = parent_child_chunks(plain)
    assert len(units) == 1
    assert units[0].child == units[0].parent  # child == parent in fallback
    assert units[0].title == "(document)"
