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

from .config import settings

# Separators tried in order, largest structural unit first.
_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


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
