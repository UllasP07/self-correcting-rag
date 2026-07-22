"""Local RAG evaluation harness (Milestone 8).

Passing tests prove the code runs; they don't prove the *answers are good*. This
harness measures answer quality with three Ragas-style metrics, each an
LLM-as-judge verdict averaged over a small labeled question set:

  faithfulness       — is the answer grounded in the retrieved context (no
                       hallucination)?
  answer_relevancy   — does the answer actually address the question?
  context_precision  — was the retrieved context relevant to the question?

No `ragas`/langchain dependency — the judges are plain LLM calls over Ollama,
and they're INJECTABLE so tests score deterministically with no model. Run it:

    python -m src.rag.eval

It prints a table and writes data/eval_report.json, which the /metrics endpoint
surfaces to Prometheus (so eval scores land on the Grafana dashboard).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

from rich.console import Console
from rich.table import Table

from .config import settings
from .llm import chat

console = Console()
_ROOT = Path(__file__).resolve().parents[2]

Judge = Callable[[str, str], bool]


def _resolve(p: str) -> Path:
    raw = Path(p)
    return raw if raw.is_absolute() else _ROOT / raw


def _yes_no(system: str, user: str) -> bool:
    return chat(system, user).strip().lower().startswith("y")


def _faithfulness_judge(answer: str, context: str) -> bool:
    return _yes_no(
        "You are a strict grader. Answer yes or no only: is EVERY claim in the "
        "answer supported by the context (no invented facts)?",
        f"Context:\n{context}\n\nAnswer:\n{answer}")


def _relevancy_judge(question: str, answer: str) -> bool:
    return _yes_no(
        "Answer yes or no only: does the answer directly address the question?",
        f"Question: {question}\n\nAnswer: {answer}")


def _precision_judge(question: str, context: str) -> bool:
    return _yes_no(
        "Answer yes or no only: is the context relevant to answering the question?",
        f"Question: {question}\n\nContext:\n{context}")


@dataclass
class EvalReport:
    scores: dict[str, float]           # metric -> mean in [0, 1]
    n: int
    items: list[dict] = field(default_factory=list)


def _context_of(answer) -> str:
    """The context an answer was built from — retrieved sections, or the SQL."""
    if answer.hits:
        return "\n".join(h.parent or h.text for h in answer.hits)
    return answer.sql or ""


def evaluate(
    dataset: list[dict],
    ask_fn: Callable | None = None,
    faithfulness_judge: Judge = _faithfulness_judge,
    relevancy_judge: Judge = _relevancy_judge,
    precision_judge: Judge = _precision_judge,
) -> EvalReport:
    """Score `dataset` (list of {"question": ...}) and return an EvalReport."""
    if ask_fn is None:
        from .pipeline import answer_question
        ask_fn = answer_question

    totals = {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_precision": 0.0}
    items: list[dict] = []
    for row in dataset:
        q = row["question"]
        ans = ask_fn(q)
        ctx = _context_of(ans)
        scored = {
            "faithfulness": float(faithfulness_judge(ans.text, ctx)),
            "answer_relevancy": float(relevancy_judge(q, ans.text)),
            "context_precision": float(precision_judge(q, ctx)),
        }
        for k, v in scored.items():
            totals[k] += v
        items.append({"question": q, "route": ans.route, **scored})

    n = len(dataset)
    scores = {k: (v / n if n else 0.0) for k, v in totals.items()}
    return EvalReport(scores=scores, n=n, items=items)


def _load_dataset() -> list[dict]:
    return json.loads(_resolve(settings.eval_set_path).read_text())


def _write_report(report: EvalReport) -> Path:
    path = _resolve(settings.eval_report_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2))
    return path


def main() -> None:
    dataset = _load_dataset()
    console.print(f"[bold]Evaluating {len(dataset)} questions…[/bold]")
    report = evaluate(dataset)

    table = Table(title="RAG eval")
    table.add_column("metric")
    table.add_column("score", justify="right")
    for metric, score in report.scores.items():
        table.add_row(metric, f"{score:.2f}")
    console.print(table)

    path = _write_report(report)
    console.print(f"[dim]wrote {path}[/dim]")


if __name__ == "__main__":
    main()
