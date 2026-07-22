"""Interactive question CLI for the thin-slice RAG.

    python -m src.rag.cli                 # interactive loop
    python -m src.rag.cli "your question"  # one-shot
"""
from __future__ import annotations

import sys

from rich.console import Console
from rich.panel import Panel

from .errors import EmbedderMismatchError, friendly_hint
from .pipeline import Answer, answer_question

console = Console()


def _handle_freeze(result: Answer) -> Answer:
    """A risky query paused at the HITL gate. Offer an inline approve/deny; if
    deferred, tell the user how to resolve it later via the admin command."""
    console.print(Panel(
        f"This query was flagged and needs approval before it runs.\n\n"
        f"SQL: {result.sql}\nflagged for: {result.risk_reason}\n"
        f"request id: {result.request_id}",
        title="🧊 Frozen — approval required", border_style="yellow"))
    choice = console.input("[yellow]approve now? (y/N, or leave for admin) [/yellow]").strip().lower()
    if choice in ("y", "yes"):
        from .graph import resume_sql
        return resume_sql(result.request_id, approved=True)
    if choice in ("n", "no"):
        from .graph import resume_sql
        return resume_sql(result.request_id, approved=False)
    console.print(f"[dim]left pending — resolve later with: "
                  f"python -m src.rag.admin approve {result.request_id}[/dim]")
    return result


def _show(question: str) -> None:
    try:
        result = answer_question(question)
        if result.status == "frozen":
            result = _handle_freeze(result)
            if result.status == "frozen":
                return  # still pending — nothing more to print
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

    # M5: show which route the question took. A data question goes to text-to-SQL
    # and has no retrieval trace — show the SQL that ran instead.
    if result.route == "structured":
        status = f" · {result.status}" if result.status != "executed" else ""
        console.print(f"[dim]route: structured (text-to-SQL){status} · {result.row_count} row(s)[/dim]")
        if result.sql:
            console.print(f"[cyan]  SQL → {result.sql}[/cyan]")
        return

    # Document route: show the CRAG self-correction trace so you can SEE the loop
    # working — did it grade the retrieval as weak, and rewrite the query?
    c = result.correction
    verdict = "relevant" if c.graded_relevant else "WEAK"
    console.print(f"[dim]route: documents · CRAG attempt(s)={c.attempts}  grade={verdict}  ({c.grade_reason})[/dim]")
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
