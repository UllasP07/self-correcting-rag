"""Interactive question CLI for the thin-slice RAG.

    python -m src.rag.cli                 # interactive loop
    python -m src.rag.cli "your question"  # one-shot
"""
from __future__ import annotations

import sys

from rich.console import Console
from rich.panel import Panel

from .pipeline import answer_question

console = Console()


def _show(question: str) -> None:
    result = answer_question(question)
    console.print(Panel(result.text, title="Answer", border_style="green"))
    # Show retrieval scores so you can SEE when retrieval is weak — this is the
    # signal CRAG will act on in Milestone 4.
    console.print(f"[dim]top similarity score: {result.top_score:.3f}[/dim]")
    for h in result.hits:
        preview = h.text[:80].replace("\n", " ")
        console.print(f"  [dim]{h.score:.3f}  {h.source}: {preview}...[/dim]")


def main() -> None:
    if len(sys.argv) > 1:
        _show(" ".join(sys.argv[1:]))
        return
    console.print("[bold]Ask a question (Ctrl-C to quit)[/bold]")
    try:
        while True:
            q = console.input("\n[cyan]?[/cyan] ").strip()
            if q:
                _show(q)
    except (KeyboardInterrupt, EOFError):
        console.print("\nbye")


if __name__ == "__main__":
    main()
