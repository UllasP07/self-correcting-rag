"""Interactive question CLI for the thin-slice RAG.

    python -m src.rag.cli                 # interactive loop
    python -m src.rag.cli "your question"  # one-shot
"""
from __future__ import annotations

import sys

from rich.console import Console
from rich.panel import Panel

from .errors import EmbedderMismatchError, friendly_hint
from .pipeline import answer_question

console = Console()


def _show(question: str) -> None:
    try:
        result = answer_question(question)
    except EmbedderMismatchError as e:
        console.print(Panel(str(e), title="Embedder mismatch", border_style="red"))
        return
    except Exception as e:  # noqa: BLE001 — show a fix-it hint for known failures
        hint = friendly_hint(e)
        if hint is None:
            raise
        console.print(Panel(hint, title="Service not ready", border_style="red"))
        return
    console.print(Panel(result.text, title="Answer", border_style="green"))
    # Show the CRAG self-correction trace so you can SEE the loop working: did it
    # grade the retrieval as weak, and did it rewrite the query and retry?
    c = result.correction
    verdict = "relevant" if c.graded_relevant else "WEAK"
    console.print(f"[dim]CRAG: attempt(s)={c.attempts}  grade={verdict}  ({c.grade_reason})[/dim]")
    if c.rewritten_query:
        console.print(f"[yellow]  ↻ rewrote query → {c.rewritten_query}[/yellow]")
    console.print(f"[dim]top score: {result.top_score:.3f}[/dim]")
    for h in result.hits:
        preview = h.text[:50].replace("\n", " ")
        section = f"[{h.title}] " if h.title else ""
        if h.rerank_score is not None:
            score_str = f"rerank={h.rerank_score:.3f} (cos={h.score:.3f})"
        else:
            score_str = f"cos={h.score:.3f}"
        console.print(f"  [dim]{score_str}  {section}{h.source}: {preview}...[/dim]")


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
