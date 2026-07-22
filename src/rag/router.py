"""Query router (Milestone 5): documents vs. structured data.

One question deserves the CRAG document loop ("what is the parental leave
policy?"); another is really a database query ("average salary in Sales?").
Sending the second down the document pipe wastes retrieval and usually fails —
the answer is in a table, not prose. So we classify first.

Two-tier, cheap → expensive (same spirit as the CRAG grader):
  1. an obvious-keyword fast path (counts/averages/"how many" → structured),
  2. otherwise an LLM classifier for the fuzzy middle.

The classifier is INJECTABLE so tests pin the routing logic without an LLM.
"""
from __future__ import annotations

import re
from typing import Callable, Literal

from .llm import chat

Route = Literal["documents", "structured"]

# Strong signals that a question wants aggregation over rows, not prose.
_STRUCTURED_HINTS = re.compile(
    r"\b(how many|count|number of|average|avg|mean|median|sum|total|"
    r"highest|lowest|most|least|per department|by department|"
    r"salary|salaries|headcount|employees in)\b",
    re.IGNORECASE,
)

ROUTE_SYSTEM = (
    "You are a query router for an enterprise assistant. Decide where a question "
    "should go:\n"
    "- 'structured': it asks for facts computable from an employee database "
    "(counts, averages, salaries, headcount, per-department figures).\n"
    "- 'documents': it asks about policies, benefits, or anything written in "
    "handbook-style prose.\n"
    "Reply with exactly one word: structured or documents."
)


def _llm_classify(question: str) -> Route:
    reply = chat(ROUTE_SYSTEM, question).strip().lower()
    return "structured" if "structured" in reply else "documents"


def route(
    question: str,
    classifier: Callable[[str], Route] | None = None,
) -> Route:
    """Return the route for `question`. `classifier` overrides the LLM (tests)."""
    if _STRUCTURED_HINTS.search(question):
        return "structured"
    classifier = classifier or _llm_classify
    return classifier(question)
