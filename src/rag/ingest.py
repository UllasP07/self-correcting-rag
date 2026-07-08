"""Ingestion pipeline (Milestone 1).

    file  ->  load_document  ->  chunk_text  ->  embed  ->  Qdrant.upsert

Run it against the ./data folder:
    python -m src.rag.ingest --recreate

`--recreate` wipes the collection first (use it when you re-tune chunking so you
don't mix old and new chunks).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

from .chunking import parent_child_chunks
from .config import settings
from .errors import friendly_hint
from .llm import embed, embedder_fingerprint
from .loaders import SUPPORTED, load_document
from .vectorstore import VectorStore

console = Console()
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _check_dim(vectors: list[list[float]]) -> None:
    """Fail early and clearly if the embedder's output dim != configured EMBED_DIM.

    Without this, a mismatch (e.g. Azure text-embedding-3-small = 1536 dims while
    EMBED_DIM=768) surfaces as a cryptic Qdrant upsert error.
    """
    if not vectors:
        return
    got = len(vectors[0])
    if got != settings.embed_dim:
        raise ValueError(
            f"Embedder returned {got}-dim vectors but EMBED_DIM={settings.embed_dim}. "
            f"Set EMBED_DIM={got} in .env and re-run with --recreate."
        )


def ingest_folder(recreate: bool = False) -> None:
    store = VectorStore()
    store.ensure_collection(recreate=recreate)

    files = [p for p in DATA_DIR.rglob("*") if p.suffix.lower() in SUPPORTED]
    if not files:
        console.print(f"[yellow]No supported files in {DATA_DIR}. "
                      f"Supported: {sorted(SUPPORTED)}[/yellow]")
        return

    total_chunks = 0
    checked = False
    for path in files:
        try:
            text = load_document(path)
        except Exception as e:  # noqa: BLE001 — surface loader errors, keep going
            console.print(f"[red]skip {path.name}: {e}[/red]")
            continue

        units = parent_child_chunks(text)  # M2: structure-aware parent-child
        if not units:
            console.print(f"[yellow]{path.name}: no text extracted[/yellow]")
            continue

        children = [u.child for u in units]   # embed + match on the small child
        parents = [u.parent for u in units]   # hand the fuller parent to the LLM
        titles = [u.title for u in units]

        vectors = embed(children)  # batch-embed all children of this file
        if not checked:
            _check_dim(vectors)  # validate once, on the first real batch
            checked = True
        n = store.upsert(children, vectors, source=path.name,
                         parents=parents, titles=titles)
        total_chunks += n
        console.print(f"[green]{path.name}[/green]: {n} chunks")

    if total_chunks:
        # Stamp the collection with the embedder that built it (guard for queries).
        store.write_fingerprint(embedder_fingerprint())

    console.print(f"\n[bold]Ingested {total_chunks} chunks "
                  f"from {len(files)} file(s).[/bold]")


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest ./data into Qdrant")
    ap.add_argument("--recreate", action="store_true",
                    help="wipe the collection before ingesting")
    args = ap.parse_args()
    try:
        ingest_folder(recreate=args.recreate)
    except Exception as e:  # noqa: BLE001 — turn known failures into guidance
        hint = friendly_hint(e)
        if hint:
            console.print(f"[red]{hint}[/red]")
            raise SystemExit(1)
        raise


if __name__ == "__main__":
    main()
