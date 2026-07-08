"""Chunking: split a long document into retrieval-sized pieces.

WHY CHUNK AT ALL?
  - Embedding models have a limited context and produce one vector per input.
    A whole 50-page PDF as one vector would be a mushy average of everything —
    useless for pinpointing the paragraph that answers a question.
  - So we split into smaller pieces, embed each, and retrieve only the pieces
    that match the query.

MILESTONE 1 uses a simple *recursive character splitter*: try to split on the
biggest natural boundary that fits (paragraph → line → sentence → word), so we
avoid cutting mid-sentence when we can. `overlap` repeats a little text between
neighbors so a fact sitting on a chunk boundary isn't lost.

In MILESTONE 2 we replace this with semantic + parent-child chunking. Keeping it
simple here lets you feel that limitation first.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .config import settings

# Separators tried in order, largest structural unit first.
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


# ---------------------------------------------------------------------------
# Milestone 2: structure-aware parent-child chunking
#
# The M1 chunker splits blindly by size, so a whole handbook collapses into a
# couple of giant blobs and retrieval can't tell one policy from another.
#
# Parent-child fixes that:
#   - CHILD  = a small, focused piece we EMBED and SEARCH on (precise matching)
#   - PARENT = the larger section it came from, which we hand to the LLM
# "Search small, answer big" — you match on a tight snippet but the model still
# gets full surrounding context.
# ---------------------------------------------------------------------------


@dataclass
class ParentChild:
    child: str    # what we embed + match on (small, precise)
    parent: str   # what the LLM reads when this child matches (fuller context)
    title: str    # section heading, for citations/display


def split_markdown_sections(text: str) -> list[tuple[str, str]]:
    """Split markdown into (title, section_text) by ATX headings (#, ##, ...).

    Text before the first heading is kept under a "(intro)" title. Each section
    keeps its heading line so the parent context is self-describing.
    """
    lines = text.splitlines()
    sections: list[tuple[str, list[str]]] = []
    current_title = "(intro)"
    current: list[str] = []
    heading = re.compile(r"^#{1,6}\s+(.*)$")

    for line in lines:
        m = heading.match(line)
        if m:
            if current:
                sections.append((current_title, current))
            current_title = m.group(1).strip()
            current = [line]
        else:
            current.append(line)
    if current:
        sections.append((current_title, current))

    return [(title, "\n".join(body).strip()) for title, body in sections
            if "\n".join(body).strip()]


def parent_child_chunks(text: str, child_size: int | None = None) -> list[ParentChild]:
    """Turn a document into parent-child units.

    Markdown → split into heading sections (the parents). Each section is then
    split into smaller children for precise embedding; every child points back
    at its full section as the parent. Non-markdown text (no headings) falls
    back to treating size-based chunks as their own parent+child.
    """
    child_size = child_size or settings.child_chunk_size
    sections = split_markdown_sections(text)

    # No markdown structure → degrade gracefully to plain chunks (child == parent).
    if len(sections) <= 1:
        return [ParentChild(child=c, parent=c, title="(document)")
                for c in chunk_text(text)]

    out: list[ParentChild] = []
    for title, section in sections:
        children = _split(section, _SEPARATORS, child_size)
        children = [c.strip() for c in children if c.strip()]
        for child in children:
            out.append(ParentChild(child=child, parent=section, title=title))
    return out


def _split(text: str, separators: list[str], chunk_size: int) -> list[str]:
    """Recursively split `text` so each piece is <= chunk_size where possible."""
    if len(text) <= chunk_size:
        return [text]

    sep = separators[0]
    rest = separators[1:]

    # Last-resort: no separator left → hard-cut by character count.
    if sep == "":
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(sep)
    chunks: list[str] = []
    buf = ""
    for part in parts:
        candidate = part if not buf else buf + sep + part
        if len(candidate) <= chunk_size:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            # A single part can still exceed chunk_size → recurse with finer sep.
            if len(part) > chunk_size:
                chunks.extend(_split(part, rest, chunk_size))
                buf = ""
            else:
                buf = part
    if buf:
        chunks.append(buf)
    return chunks


def _add_overlap(chunks: list[str], overlap: int) -> list[str]:
    """Prepend the tail of the previous chunk to each chunk (sliding context)."""
    if overlap <= 0 or len(chunks) <= 1:
        return chunks
    out = [chunks[0]]
    for i in range(1, len(chunks)):
        tail = chunks[i - 1][-overlap:]
        out.append(tail + " " + chunks[i])
    return out


def chunk_text(text: str) -> list[str]:
    """Public API: split cleaned document text into overlapping chunks."""
    text = text.strip()
    if not text:
        return []
    raw = _split(text, _SEPARATORS, settings.chunk_size)
    raw = [c.strip() for c in raw if c.strip()]
    return _add_overlap(raw, settings.chunk_overlap)
