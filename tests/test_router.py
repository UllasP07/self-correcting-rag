"""Tests for the query router (documents vs. structured)."""
from src.rag.router import route


def test_keyword_fast_path_routes_structured():
    # These hit the keyword pre-filter, so the classifier must NOT be consulted.
    boom = lambda q: (_ for _ in ()).throw(AssertionError("classifier called"))
    assert route("How many employees are in Engineering?", classifier=boom) == "structured"
    assert route("What is the average salary by department?", classifier=boom) == "structured"


def test_defers_to_classifier_when_no_keyword():
    calls = []
    classifier = lambda q: calls.append(q) or "documents"
    assert route("What is our parental leave policy?", classifier=classifier) == "documents"
    assert calls == ["What is our parental leave policy?"]


def test_classifier_can_route_structured():
    assert route("Tell me about the team head&size", classifier=lambda q: "structured") == "structured"
