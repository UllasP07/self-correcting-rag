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

from .chunking import chunk_text
from .loaders import SUPPORTED, load_document
from .ollama_client import embed
from .vectorstore import VectorStore

console = Console()
DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def ingest_folder(recreate: bool = False) -> None:
    store = VectorStore()
    store.ensure_collection(recreate=recreate)

    files = [p for p in DATA_DIR.rglob("*") if p.suffix.lower() in SUPPORTED]
    if not files:
        console.print(f"[yellow]No supported files in {DATA_DIR}. "
                      f"Supported: {sorted(SUPPORTED)}[/yellow]")
        return

    total_chunks = 0
    for path in files:
        try:
            text = load_document(path)
        except Exception as e:  # noqa: BLE001 — surface loader errors, keep going
            console.print(f"[red]skip {path.name}: {e}[/red]")
            continue

        chunks = chunk_text(text)
        if not chunks:
            console.print(f"[yellow]{path.name}: no text extracted[/yellow]")
            continue

        vectors = embed(chunks)  # batch-embed all chunks of this file
        n = store.upsert(chunks, vectors, source=path.name)
        total_chunks += n
        console.print(f"[green]{path.name}[/green]: {n} chunks")

    console.print(f"\n[bold]Ingested {total_chunks} chunks "
                  f"from {len(files)} file(s).[/bold]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ingest ./data into Qdrant")
    ap.add_argument("--recreate", action="store_true",
                    help="wipe the collection before ingesting")
    args = ap.parse_args()
    ingest_folder(recreate=args.recreate)
