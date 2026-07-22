"""Admin approval CLI for frozen SQL requests (Milestone 6).

When a risky query freezes, its state is checkpointed to disk and a row is added
to the approvals registry. This command lets an admin — possibly in a different
terminal, minutes later, after a restart — review and resolve those requests:

    python -m src.rag.admin list                 # show what's waiting
    python -m src.rag.admin approve <request_id>  # resume + execute
    python -m src.rag.admin deny <request_id>     # resume + refuse

Approve/deny resumes the persisted graph from exactly where it paused — that's
the "persistent state" the milestone is about.
"""
from __future__ import annotations

import sys

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import approvals
from .graph import resume_sql

console = Console()


def _list() -> None:
    pending = approvals.list_pending()
    if not pending:
        console.print("[dim]No pending approvals.[/dim]")
        return
    table = Table(title="Pending SQL approvals")
    table.add_column("request_id", style="cyan")
    table.add_column("question")
    table.add_column("SQL", style="yellow")
    table.add_column("flagged for", style="red")
    table.add_column("created", style="dim")
    for r in pending:
        table.add_row(r.request_id, r.question, r.sql, r.reasons, r.created_at)
    console.print(table)


def _resolve(request_id: str, approved: bool) -> None:
    rec = approvals.get(request_id)
    if rec is None:
        console.print(f"[red]No such request: {request_id}[/red]")
        return
    if rec.status != "pending":
        console.print(f"[yellow]{request_id} is already {rec.status}.[/yellow]")
        return
    result = resume_sql(request_id, approved=approved)
    verb = "APPROVED" if approved else "DENIED"
    color = "green" if approved else "red"
    console.print(f"[{color}]{verb}[/{color}] {request_id}: {result.sql}")
    console.print(Panel(result.text, title="Result", border_style=color))


def main() -> None:
    args = sys.argv[1:]
    cmd = args[0] if args else "list"
    if cmd == "list":
        _list()
    elif cmd in ("approve", "deny") and len(args) == 2:
        _resolve(args[1], approved=(cmd == "approve"))
    else:
        console.print("usage: python -m src.rag.admin [list | approve <id> | deny <id>]")


if __name__ == "__main__":
    main()
